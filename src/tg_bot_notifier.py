"""
Модуль для отправки уведомлений в Telegram.

Содержит класс TelegramBotNotifier, который обертывает библиотеку `telebot`
для упрощенной отправки сообщений, используя параметры из переменных окружения.
"""
import logging
import os
from typing import Optional

import telebot
from dotenv import load_dotenv

load_dotenv()


class TelegramBotNotifier:
    """Класс для отправки уведомлений через библиотеку telebot с поддержкой .env."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """Инициализация бота.

        Пытается загрузить токен и ID чата из аргументов. Если они не переданы,
        ищет переменные окружения `TG_BOT_TOKEN` и `TG_CHAT_ID`.

        Args:
            token (Optional[str]): Токен Telegram-бота.
            chat_id (Optional[str]): ID чата (или пользователя) для отправки сообщений.
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
        """Отправляет текстовое сообщение в Telegram.

        Поддерживает HTML-разметку. Ошибки при отправке логируются, но не
        прерывают выполнение программы.

        Args:
            message (str): Текст сообщения (может содержать HTML-теги).
        """
        if not self.bot or not self.chat_id:
            return

        try:
            # parse_mode='HTML' позволяет использовать теги <b>, <i>, <code>
            self.bot.send_message(self.chat_id, message, parse_mode="HTML")
        except Exception as e:
            self.logger.error(f"Не удалось отправить сообщение в Telegram: {e}")
