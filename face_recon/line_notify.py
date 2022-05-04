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


# if __name__ == "__main__":
#     token = 'X8TD43e49jlLWDMFGDLD3ATshAHSM3tn4h1SDU0O2Bg'
#     message = '基本功能測試'
#     lineNotifyMessage(token, message)
