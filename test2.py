import base64
import requests

ESP8266URL = "http://192.168.4.1/"


def fetch_view_serial():
    try:
        response = requests.get(f"{ESP8266URL}/viewserial")
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        # print("Received /viewserial:", json.dumps(data, indent=2))
        return data
    except requests.RequestException as e:
        print(f"Error fetching /viewserial: {e}")
        return None


def view_client_data():
    try:
        response = requests.get(f"{ESP8266URL}/viewclientdata")
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        # print("Received /viewclientdata:", json.dumps(data, indent=2))
        return data
    except requests.RequestException as e:
        print(f"Error fetching /viewclientdata: {e}")
        return None


def send_server_data(song_id):
    try:
        payload = {"id": song_id}
        response = requests.post(f"{ESP8266URL}/serverdata", json=payload)
        response.raise_for_status()
        # print("Sent /serverdata:", json.dumps(payload, indent=2))
        # print("Response:", response.json())
    except requests.RequestException as e:
        print(f"Error sending to /serverdata: {e}")


def clear_client_data():
    try:
        payload = {"code": 0}
        response = requests.post(f"{ESP8266URL}/clientdata", json=payload)
        response.raise_for_status()
        # print("Sent /clientdata:", json.dumps(payload, indent=2))
        # print("Response:", response.json())
    except requests.RequestException as e:
        print(f"Error sending to /clientdata: {e}")


def main():
    # Step 1: Fetch /viewserial
    fetch_view_serial()

    # Step 2: Send {"id": 47} to /serverdata
    send_server_data(47)

    # Step 3: Send {"code": 0} to /clientdata
    clear_client_data()


if __name__ == "__main__":
    main()
