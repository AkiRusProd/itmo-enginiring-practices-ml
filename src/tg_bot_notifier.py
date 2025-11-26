import logging
import os
from typing import Optional

import telebot
from dotenv import load_dotenv

load_dotenv()


class TelegramBotNotifier:
    """Класс для отправки уведомлений через библиотеку telebot с поддержкой .env."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Инициализация бота.
        Если token или chat_id не переданы, пытается взять их из переменных окружения.
        """
        self.logger = logging.getLogger("tg_notifier")

        self.token = token or os.getenv("TG_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TG_CHAT_ID")

        self.bot = None

        if self.token and self.chat_id:
            try:
                self.bot = telebot.TeleBot(self.token)
            except Exception as e:
                self.logger.error(f"Ошибка инициализации бота: {e}")
        else:
            self.logger.warning(
                "⚠️ TG_BOT_TOKEN или TG_CHAT_ID не найдены в .env или аргументах. Уведомления отключены."
            )

    def send_message(self, message: str) -> None:
        """
        Отправка сообщения. Ошибки глушатся, чтобы не ломать основной пайплайн.
        """
        if not self.bot or not self.chat_id:
            return

        try:
            # parse_mode='HTML' позволяет использовать теги <b>, <i>, <code>
            self.bot.send_message(self.chat_id, message, parse_mode="HTML")
        except Exception as e:
            self.logger.error(f"Не удалось отправить сообщение в Telegram: {e}")
