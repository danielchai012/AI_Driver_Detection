import requests


def lineNotifyMessage(msg):
    token = 'X8TD43e49jlLWDMFGDLD3ATshAHSM3tn4h1SDU0O2Bg'
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    payload = {'message': msg}
    r = requests.post("https://notify-api.line.me/api/notify",
                      headers=headers, params=payload)
    return r.status_code


def lineNotifyMessage_Pic(msg, img_path):
    token = 'X8TD43e49jlLWDMFGDLD3ATshAHSM3tn4h1SDU0O2Bg'
    headers = {
        "Authorization": "Bearer " + token
    }
    print(img_path)
    data = {'message': msg
            }
    image = open(img_path, 'rb')
    imageFile = {'imageFile': image}
    r = requests.post("https://notify-api.line.me/api/notify",
                      headers=headers, data=data, files=imageFile)
    return r.status_code

# if __name__ == "__main__":
#     token = 'X8TD43e49jlLWDMFGDLD3ATshAHSM3tn4h1SDU0O2Bg'
#     message = '基本功能測試'
#     lineNotifyMessage(token, message)
