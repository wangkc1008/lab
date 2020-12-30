"""
created by PyCharm
date: 2020/7/11
time: 10:06
user: wkc
"""

from aip import AipOcr

# 认证信息
APP_ID = ''
API_KEY = '91230a3466df42ee92ab9b5abd5e356f'
SECRET_KEY = '503e19f741f24a6d9552fe573a7bffb8'


def get_ocr_str(file_path, origin_format=True):
    """
    图片转文字
    :param file_path: 图片路径
    :return:
    """
    with open(file_path, 'rb') as fp:
        file_bytes = fp.read()
    return get_ocr_str_from_bytes(file_bytes, origin_format)


def get_ocr_str_from_bytes(file_bytes, origin_format=True):
    """
    图片转文字
    :param file_bytes: 图片的字节
    :return:
    """
    options = {
        'detect_direction': 'false',
        'language_type': 'CHN_ENG',
    }
    ocr = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    result_dict = ocr.basicGeneral(file_bytes, options)
    if origin_format:
        result_str = '\n'.join([entity['words'] for entity in result_dict['words_result']])
    else:
        result_str = ''.join([entity['words'] for entity in result_dict['words_result']])
    return result_str


if __name__ == '__main__':
    IMAGE_PATH = "./image1.jpg"
    text = get_ocr_str(IMAGE_PATH)
    with open("./image1.txt", "w", encoding="utf-8") as f:  # 将识别出来的文字存到本地
        # print(text)
        f.write(str(text))

