import telebot
import os
import cv2
import numpy as np
import sunglasses
from googletrans import Translator

token = os.getenv("GLASSES_TOKEN")
bot = telebot.TeleBot(token)
user_states = {}


def exception_catcher(base_function):
    def new_function(*args,
                     **kwargs):  # This allows you to decorate functions without worrying about what arguments they take
        try:
            return base_function(*args, **kwargs)  # base_function is whatever function this decorator is applied to
        except Exception as e:
            err_msg = base_function.__name__ + ' => ' + str(e)
            print(err_msg)

    return new_function


@exception_catcher
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_chat_action(message.chat.id, 'typing')
    bot.reply_to(message, f'Я твой личный помощник. Приятно познакомиться, {message.from_user.first_name}. '
                          f'Для ознакомления с функционалом выполни /help')


@exception_catcher
@bot.message_handler(commands=['help'])
def send_help(message):
    bot.send_chat_action(message.chat.id, 'typing')
    bot.reply_to(message, '/start - начать\n'
                          '/send - отправить селфи')


@exception_catcher
@bot.message_handler(commands=['send'])
def send_photo(message):
    user_states[message.from_user.id] = 1
    bot.register_next_step_handler(message, get_media_message)
    bot.reply_to(message, 'Сделай селфи и старайся смотреть ровно в камеру :)')


@exception_catcher
@bot.message_handler(content_types=['photo'])
def get_media_message(message):
    if message.photo is None or user_states.get(message.from_user.id) != 1:
        return

    user_states.pop(message.from_user.id, None)
    max_size_ctr = -1
    max_size = 0
    photo_ctr = 0
    for photo_size in message.photo:
        if photo_size.width * photo_size.height > max_size:
            max_size = photo_size.width * photo_size.height
            max_size_ctr = photo_ctr

        photo_ctr += 1

    if max_size_ctr < 0:
        return

    photo_size = message.photo[max_size_ctr]
    file_info = bot.get_file(photo_size.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    photo = cv2.imdecode(np.frombuffer(downloaded_file, dtype=np.uint8), 1)
    shape_text, result_image = sunglasses.process_image(photo)
    if result_image is None:
        return

    result_image_str = cv2.imencode('.jpg', result_image)

    translator = Translator()
    shape_text_ru = translator.translate(shape_text, src='en', dest='ru')

    bot.reply_to(message, f'Похоже форма твоего лица: {shape_text_ru.text}')
    bot.send_photo(chat_id=message.chat.id,
                   photo=result_image_str[1].tobytes(),
                   caption='Взгляни, что я тебе подобрал!',
                   reply_to_message_id=message.id)


bot.polling(none_stop=True)
