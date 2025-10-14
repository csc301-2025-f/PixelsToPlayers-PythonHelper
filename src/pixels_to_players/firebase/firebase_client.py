import requests


class FirebaseClient:
    """
    A utility class to handle Firebase related operations, including authentication,
    data fetching, and data modification through Firebase REST API.
    """

    def __init__(self, api_key: str, database_url: str):
        """
        Initialize the Firebase client with the API key and database URL.

        :param api_key: Firebase API key for authentication
        :param database_url: URL for the Firebase Realtime Database
        """
        self.api_key = api_key
        self.database_url = database_url
        self.id_token = None  # Stores the authenticated user's token

    def authenticate_user(self, email: str, password: str) -> bool:
        """
        Authenticate a user with email and password using the Firebase REST API.

        :param email: User's email address
        :param password: User's password
        :return: True if authentication is successful, False otherwise
        """
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.api_key}"
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            self.id_token = response.json().get('idToken')
            return True
        else:
            print(f"Authentication failed: {response.json()}")
            return False

    def get_data(self, path: str):
        """
        Fetches data from the specified path in the Firebase Realtime Database.

        :param path: Path in the Firebase database (e.g., "users/user1")
        :return: Fetched data as a dictionary, or None if the request fails
        """
        if not self.id_token:
            raise Exception("User is not authenticated.")
        url = f"{self.database_url}/{path}.json?auth={self.id_token}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data: {response.json()}")
            return None

    def set_data(self, path: str, data: dict) -> bool:
        """
        Writes data to the specified path in the Firebase Realtime Database.

        :param path: Path in the Firebase database (e.g., "users/user1")
        :param data: Data to be written (as a dictionary)
        :return: True if the write operation is successful, False otherwise
        """
        if not self.id_token:
            raise Exception("User is not authenticated.")
        url = f"{self.database_url}/{path}.json?auth={self.id_token}"
        response = requests.put(url, json=data)
        if response.status_code == 200:
            return True
        else:
            print(f"Failed to set data: {response.json()}")
            return False

    def append_data(self, path: str, data: dict) -> bool:
        """
        Appends data to an existing path in the Firebase database. This performs a 'push' operation.

        :param path: Path in the Firebase database (e.g., "users")
        :param data: Data to be appended (as a dictionary)
        :return: True if the append operation is successful, False otherwise
        """
        if not self.id_token:
            raise Exception("User is not authenticated.")
        url = f"{self.database_url}/{path}.json?auth={self.id_token}"
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return True
        else:
            print(f"Failed to append data: {response.json()}")
            return False

    def update_data(self, path: str, data: dict) -> bool:
        """
        Updates specific fields in an existing path in the Firebase database.

        :param path: Path in the Firebase database (e.g., "users/user1")
        :param data: Data fields to update (as a dictionary)
        :return: True if the update operation is successful, False otherwise
        """
        if not self.id_token:
            raise Exception("User is not authenticated.")
        url = f"{self.database_url}/{path}.json?auth={self.id_token}"
        response = requests.patch(url, json=data)
        if response.status_code == 200:
            return True
        else:
            print(f"Failed to update data: {response.json()}")
            return False


if __name__ == "__main__":
    # Example usage of the FirebaseClient (for testing purposes)
    # Replace `YOUR_API_KEY` and `YOUR_DATABASE_URL` with actual values
    firebase = FirebaseClient(
        api_key="YOUR_API_KEY",
        database_url="https://your-database-name.firebaseio.com"
    )

    # Test authentication
    if firebase.authenticate_user("email@example.com", "password123"):
        print("Authentication successful.")

        # Fetch data example
        data = firebase.get_data("path/to/data")
        print("Fetched data:", data)

        # Write data example
        if firebase.set_data("path/to/data", {"key": "value"}):
            print("Data written successfully.")

        # Append data example
        if firebase.append_data("path/to/data", {"new_key": "new_value"}):
            print("Data appended successfully.")
