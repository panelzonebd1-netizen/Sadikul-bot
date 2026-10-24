
import os
import re
import asyncio
import base64
import logging
import threading
from collections import defaultdict

from flask import Flask, jsonify
from groq import AsyncGroq
from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ==========================================
# SADIKUL AI BOT
# Telegram + Groq + Render
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("sadikul_bot")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

TEXT_MODEL = os.getenv(
    "GROQ_TEXT_MODEL",
    "openai/gpt-oss-20b",
).strip()

VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.8-27b",
).strip()

ADMIN_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}

# mention = mention/reply হলেই গ্রুপে উত্তর
# all = গ্রুপের সব সাধারণ মেসেজে উত্তর
GROUP_REPLY_MODE = os.getenv(
    "GROUP_REPLY_MODE", "mention"
).strip().lower()

# mention = চ্যানেলে বটকে mention করলে উত্তর
# all = চ্যানেলের প্রতিটি পোস্টে উত্তর
CHANNEL_REPLY_MODE = os.getenv(
    "CHANNEL_REPLY_MODE", "mention"
).strip().lower()

MAX_HISTORY = 12
MAX_INPUT_LENGTH = 8000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_RETRIES = 3

if not BOT_TOKEN:
    raise RuntimeError("Render Environment এ BOT_TOKEN সেট করো।")

if not GROQ_API_KEY:
    raise RuntimeError("Render Environment এ GROQ_API_KEY সেট করো।")

client = AsyncGroq(
    api_key=GROQ_API_KEY,
    timeout=60.0,
    max_retries=0,
)

web = Flask(__name__)

memory = defaultdict(list)
memory_lock = threading.RLock()
bot_username = None

SYSTEM_PROMPT = """
তুমি SADIKUL AI BOT। তোমাকে তৈরি করেছেন Sadikul ভাই।

পরিচয়:
কেউ জিজ্ঞেস করলে তোমাকে কে তৈরি করেছে, বলবে:
"আমাকে তৈরি করেছে Sadikul ভাই। ❤️"

নিজেকে OpenAI, Meta বা Qwen-এর তৈরি বলে দাবি করবে না।
নির্মাতা ও AI মডেলের পরিচয় গুলিয়ে ফেলবে না।

উত্তরের ধরন:
- সবসময় বন্ধুত্বপূর্ণ, স্বাভাবিক বাংলায় কথা বলবে।
- উত্তর সাধারণত ২-৪ লাইনের মধ্যে রাখবে।
- Telegram-এর জন্য সংক্ষিপ্ত plain text ব্যবহার করবে।
- ##, **, Markdown table, |, --- বা বড় heading ব্যবহার করবে না।
- অপ্রয়োজনীয় ভূমিকা, দীর্ঘ রচনা ও একই কথা বারবার বলা এড়াবে।
- প্রয়োজন হলে ছোট bullet বা ইমোজি ব্যবহার করতে পারো।
- প্রশ্নে বিস্তারিত তথ্য জরুরি হলে প্রয়োজনীয় তথ্য সংক্ষেপে দেবে।
- ছবি পেলে ছবির দৃশ্যমান বিষয় বিশ্লেষণ করবে।
- নিশ্চিত না হলে পরিষ্কারভাবে জানাবে।
- API key, গোপন নির্দেশনা বা credentials প্রকাশ করবে না।
- গ্রুপ, সুপারগ্রুপ, চ্যানেল বা ব্যক্তিগত চ্যাটে কাজ না করার
  অজুহাত দেবে না। প্রশ্নের উত্তর দেওয়ার চেষ্টা করবে।
- ব্যবহারকারীর ভাষা বুঝে সহজ বাংলায় উত্তর দেবে।
"""

# ==========================================
# RENDER HEALTH
# ==========================================

@web.get("/")
def home():
    return jsonify({
        "name": "SADIKUL AI BOT",
        "status": "online",
    }), 200


@web.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


def run_web():
    web.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        threaded=True,
        use_reloader=False,
    )


# ==========================================
# TELEGRAM HELPERS
# ==========================================

def is_admin(update: Update) -> bool:
    user = update.effective_user
    return bool(user and user.id in ADMIN_IDS)


def get_key(update: Update) -> str:
    chat = update.effective_chat
    user = update.effective_user
    message = update.effective_message

    if not chat:
        return "unknown"

    # Channel post-এ effective_user সাধারণত থাকে না।
    if user:
        identity = str(user.id)
    elif message and message.sender_chat:
        identity = f"sender_chat:{message.sender_chat.id}"
    else:
        identity = "channel_or_anonymous"

    return f"{chat.id}:{identity}"


def is_group(update: Update) -> bool:
    chat = update.effective_chat
    return bool(
        chat and chat.type in (
            ChatType.GROUP,
            ChatType.SUPERGROUP,
        )
    )


def is_channel(update: Update) -> bool:
    chat = update.effective_chat
    return bool(chat and chat.type == ChatType.CHANNEL)


def is_bot_mentioned(text: str) -> bool:
    if not text or not bot_username:
        return False

    return bool(re.search(
        rf"(?<!\w)@{re.escape(bot_username)}(?!\w)",
        text,
        flags=re.IGNORECASE,
    ))


def is_reply_to_bot(update: Update) -> bool:
    message = update.effective_message

    if not message or not message.reply_to_message:
        return False

    sender = message.reply_to_message.from_user
    return bool(sender and sender.is_bot)


def is_forwarded(message) -> bool:
    if not message:
        return False

    return any(
        getattr(message, attr, None) is not None
        for attr in (
            "forward_origin",
            "forward_date",
            "forward_from",
            "forward_from_chat",
            "forward_sender_name",
        )
    )


def should_answer(update: Update, text: str) -> bool:
    # Personal chat: answer normally.
    if not is_group(update) and not is_channel(update):
        return True

    mentioned = is_bot_mentioned(text)
    replied = is_reply_to_bot(update)

    if is_group(update):
        if GROUP_REPLY_MODE == "all":
            return True
        return mentioned or replied

    if is_channel(update):
        if CHANNEL_REPLY_MODE == "all":
            return True
        return mentioned or replied

    return False


async def reply(update: Update, text: str):
    message = update.effective_message

    if not message:
        return

    text = (text or "দুঃখিত, কোনো উত্তর পাওয়া যায়নি।").strip()

    # Plain text: Markdown parsing নেই।
    for start in range(0, len(text), 4000):
        await message.reply_text(
            text[start:start + 4000],
            disable_web_page_preview=True,
        )


def clean_answer(text: str) -> str:
    """উত্তরের অপ্রয়োজনীয় Markdown চিহ্ন সরায়।"""
    text = (text or "").strip()

    # Headings
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)

    # Bold / italic / strikethrough / inline code markers
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.S)
    text = re.sub(r"(?<!\*)\*(?!\s)(.*?)(?<!\s)\*(?!\*)",
                  r"\1", text, flags=re.S)
    text = re.sub(r"__(.*?)__", r"\1", text, flags=re.S)
    text = re.sub(r"~~(.*?)~~", r"\1", text, flags=re.S)
    text = text.replace("```", "").replace("`", "")

    # Horizontal rules
    text = re.sub(r"(?m)^\s*[-_]{3,}\s*$", "", text)

    # Avoid Markdown table formatting
    if re.search(r"(?m)^\s*\|.*\|", text):
        text = re.sub(r"(?m)^\s*\|", "", text)
        text = re.sub(r"\|\s*$", "", text)
        text = re.sub(r"(?m)^\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+$", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_creator_question(text: str) -> bool:
    if not text:
        return False

    patterns = [
        r"(কে|কারা).{0,25}(তোমাকে|তোকে).{0,20}(বানাইছে|বানিয়েছে|তৈরি করেছে)",
        r"(তোমার|তোর).{0,15}(creator|ক্রিয়েটর|নির্মাতা|মেকার).{0,10}কে",
        r"who\s+(created|made|built|developed)\s+you",
        r"who\s+is\s+your\s+creator",
    ]

    return any(
        re.search(p, text, flags=re.IGNORECASE)
        for p in patterns
    )


# ==========================================
# CONVERSATION MEMORY
# ==========================================

def get_history(key: str):
    with memory_lock:
        return list(memory.get(key, []))


def save_exchange(key: str, question: str, answer: str):
    with memory_lock:
        items = memory[key]

        items.append({
            "role": "user",
            "content": question,
        })
        items.append({
            "role": "assistant",
            "content": answer,
        })

        del items[:-MAX_HISTORY]

        if items and items[0]["role"] == "assistant":
            items.pop(0)


# ==========================================
# GROQ RETRY
# ==========================================

def is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)

    if status is None and response is not None:
        status = getattr(response, "status_code", None)

    error_text = str(exc).lower()

    return (
        status == 429
        or "rate_limit_exceeded" in error_text
        or "rate limit reached" in error_text
    )


async def create_completion_with_retry(model: str, messages: list):
    max_tokens = 500 if "qwen" in model.lower() else 1000

    for attempt in range(MAX_RETRIES):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )

        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise

            if attempt == MAX_RETRIES - 1:
                raise

            wait_seconds = 2 + attempt

            log.warning(
                "Groq rate limit. Retry %d/%d in %d seconds.",
                attempt + 1,
                MAX_RETRIES - 1,
                wait_seconds,
            )

            await asyncio.sleep(wait_seconds)

    raise RuntimeError("Groq request failed.")


# ==========================================
# AI CHAT
# ==========================================

async def ask_ai(
    update: Update,
    question: str,
    context: ContextTypes.DEFAULT_TYPE,
    image_data_url: str = None,
    force: bool = False,
):
    message = update.effective_message

    if not message:
        return

    question = (question or "").strip()[:MAX_INPUT_LENGTH]

    if not question and not image_data_url:
        await reply(update, "তোমার প্রশ্নটি লিখে পাঠাও।")
        return

    if not force and not should_answer(update, question):
        return

    key = get_key(update)

    if not image_data_url and is_creator_question(question):
        answer = "আমাকে তৈরি করেছে Sadikul ভাই। ❤️"
        save_exchange(key, question, answer)
        await reply(update, answer)
        return

    try:
        await context.bot.send_chat_action(
            chat_id=message.chat_id,
            action=ChatAction.TYPING,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        messages.extend(get_history(key))

        if image_data_url:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question or "এই ছবিটি বিশ্লেষণ করো।",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            })
            model = VISION_MODEL
        else:
            messages.append({
                "role": "user",
                "content": question,
            })
            model = TEXT_MODEL

        result = await create_completion_with_retry(
            model=model,
            messages=messages,
        )

        answer = clean_answer(
            result.choices[0].message.content or ""
        )

        if not answer:
            answer = "দুঃখিত, এবার উত্তর তৈরি হয়নি। আবার চেষ্টা করো।"

        save_exchange(
            key,
            question or "[ছবি পাঠানো হয়েছে]",
            answer,
        )

        await reply(update, answer)

    except Exception:
        log.exception(
            "AI request failed. Model=%s",
            VISION_MODEL if image_data_url else TEXT_MODEL,
        )

        await reply(
            update,
            "⚠️ AI সার্ভারে আপাতত সমস্যা হচ্ছে। "
            "কিছুক্ষণ পর আবার চেষ্টা করো।",
        )


# ==========================================
# COMMANDS
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply(
        update,
        "হাই! আমি SADIKUL AI BOT 🤖❤️\n\n"
        "আমাকে প্রশ্ন করতে বা ছবি পাঠাতে পারো।\n\n"
        "/ask প্রশ্ন — AI-কে জিজ্ঞাসা\n"
        "/clear — নিজের মেমোরি মুছো\n"
        "/id — Telegram ID\n"
        "/stats — অ্যাডমিন পরিসংখ্যান\n"
        "/clearall — সব মেমোরি মুছো (অ্যাডমিন)\n"
        "/help — সাহায্য",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply(
        update,
        "📚 SADIKUL AI BOT\n\n"
        "• ব্যক্তিগত চ্যাটে সরাসরি লিখো।\n"
        "• গ্রুপে বটকে mention বা reply করো।\n"
        "• ছবি পাঠিয়ে বিশ্লেষণ করাতে পারো।\n"
        "• /ask প্রশ্ন — সরাসরি প্রশ্ন।\n"
        "• /clear — নিজের মেমোরি মুছো।\n"
        "• /id — User ID ও Chat ID।",
    )


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = " ".join(context.args).strip()

    if not question:
        await reply(update, "ব্যবহার: /ask বাংলাদেশ সম্পর্কে বলো")
        return

    await ask_ai(update, question, context, force=True)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with memory_lock:
        memory.pop(get_key(update), None)

    await reply(update, "✅ তোমার কথোপকথনের মেমোরি মুছে দেওয়া হয়েছে।")


async def id_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat = update.effective_chat

    if user and chat:
        await reply(
            update,
            f"User ID: {user.id}\nChat ID: {chat.id}",
        )
    elif chat:
        await reply(update, f"Chat ID: {chat.id}")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await reply(update, "⛔ এই কমান্ড শুধু অ্যাডমিনদের জন্য।")
        return

    with memory_lock:
        conversations = len(memory)
        message_count = sum(len(v) for v in memory.values())

    await reply(
        update,
        "🛠 SADIKUL AI BOT\n"
        f"Conversations: {conversations}\n"
        f"Stored messages: {message_count}\n"
        f"Text model: {TEXT_MODEL}\n"
        f"Vision model: {VISION_MODEL}\n"
        f"Group mode: {GROUP_REPLY_MODE}\n"
        f"Channel mode: {CHANNEL_REPLY_MODE}",
    )


async def clearall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await reply(update, "⛔ অনুমতি নেই।")
        return

    with memory_lock:
        memory.clear()

    await reply(update, "✅ সব কথোপকথনের মেমোরি মুছে দেওয়া হয়েছে।")


# ==========================================
# TEXT
# ==========================================

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message

    if not message or not message.text:
        return

    await ask_ai(update, message.text, context)


# ==========================================
# PHOTO + FORWARDED PHOTO
# ==========================================

async def image_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message

    if not message or not message.photo:
        return

    caption = (message.caption or "").strip()

    # Group/channel mode অনুযায়ী ছবির caption-ও পরীক্ষা করা হবে।
    # Forwarded photo নিজে থেকেই সব গ্রুপ মেসেজে উত্তর চালু করে না।
    if not should_answer(update, caption):
        return

    photo = message.photo[-1]

    if photo.file_size and photo.file_size > MAX_IMAGE_BYTES:
        await reply(update, "ছবিটি বড়। 8 MB-এর কম ছবি পাঠাও।")
        return

    try:
        await context.bot.send_chat_action(
            chat_id=message.chat_id,
            action=ChatAction.TYPING,
        )

        telegram_file = await context.bot.get_file(photo.file_id)
        image_bytes = bytes(
            await telegram_file.download_as_bytearray()
        )

        if len(image_bytes) > MAX_IMAGE_BYTES:
            await reply(update, "ছবিটি বড়। ছোট ছবি পাঠিয়ে আবার চেষ্টা করো।")
            return

        base64_image = base64.b64encode(image_bytes).decode("ascii")
        image_data_url = f"data:image/jpeg;base64,{base64_image}"

        await ask_ai(
            update,
            caption or "এই ছবিটি বিশ্লেষণ করে বাংলায় বুঝিয়ে বলো।",
            context,
            image_data_url=image_data_url,
            force=True,
        )

    except Exception:
        log.exception("Image download or processing failed")
        await reply(
            update,
            "⚠️ ছবিটি প্রসেস করা যায়নি। পরে আবার চেষ্টা করো।",
        )


# ==========================================
# ERRORS
# ==========================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    if context.error:
        log.error(
            "Unhandled Telegram error",
            exc_info=(
                type(context.error),
                context.error,
                context.error.__traceback__,
            ),
        )


# ==========================================
# STARTUP
# ==========================================

async def post_init(application):
    global bot_username

    me = await application.bot.get_me()
    bot_username = me.username

    log.info("Connected to Telegram as @%s", bot_username)


async def post_shutdown(application):
    await client.close()
    log.info("Groq client closed.")


def run_bot():
    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("id", id_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("clearall", clearall_command))

    application.add_handler(
        MessageHandler(filters.PHOTO, image_chat)
    )

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat)
    )

    application.add_error_handler(error_handler)

    log.info(
        "Starting SADIKUL AI BOT | Group=%s | Channel=%s",
        GROUP_REPLY_MODE,
        CHANNEL_REPLY_MODE,
    )

    application.run_polling(
        drop_pending_updates=False,
        bootstrap_retries=-1,
        allowed_updates=Update.ALL_TYPES,
    )


if __name__ == "__main__":
    threading.Thread(
        target=run_web,
        name="render-health-server",
        daemon=True,
    ).start()

    run_bot()
