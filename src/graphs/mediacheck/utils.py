
import os
import requests


class RedditClient:

    def __init__(self):
        # Load environment variables
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv(
            'REDDIT_USER_AGENT', 'news_agent by u/gerim_dealer')
        self.refresh_token = os.getenv('REDDIT_REFRESH_TOKEN', "")
        self.base_url = 'https://oauth.reddit.com/r'

        # Get access token using refresh token
        self.access_token = self._get_access_token()

    def _get_access_token(self):
        """
        Use the refresh token to get a new access token from Reddit
        """
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        headers = {
            'User-Agent': self.user_agent,
        }
        data = {
            'grant_type': 'client_credentials',
            'device_id': 'DO_NOT_TRACK_THIS_DEVICE'
        }


        response = requests.post(
            'https://www.reddit.com/api/v1/access_token',
            auth=auth,
            headers=headers,
            data=data
        )

        if response.status_code == 200:
            return response.json()['access_token']
        else:
            raise Exception(f"Failed to refresh access token: {response.text}")

    def get_subreddit_posts(self, subreddit, limit=10, sort='new'):
        """
        Retrieve posts from a specified subreddit
        """
        headers = {
            'Authorization': f'bearer {self.access_token}',
            'User-Agent': self.user_agent
        }
        url = f'{self.base_url}/{subreddit}/{sort}.json'
        params = {
            'limit': limit,
            'raw_json': 1
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                posts = response.json()['data']['children']
                return [self._format_post(post['data']) for post in posts]
            else:
                print(f"Error fetching posts: {response.status_code}")
                return []
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return []

    def _format_post(self, post_data):
        """
        Format raw Reddit post data into structured dictionary
        """
        return {
            'title': post_data.get('title', ''),
            'author': post_data.get('author', ''),
            'url': post_data.get('url', ''),
            'score': post_data.get('score', 0),
            'num_comments': post_data.get('num_comments', 0),
            'created_utc': post_data.get('created_utc', 0),
            'media': post_data.get('media', None),
            'is_video': post_data.get('is_video', False),
            'permalink': f"https://reddit.com{post_data.get('permalink', '')}",
            'thumbnail': post_data.get('thumbnail', None),
            'selftext_html': post_data.get('selftext_html', None)
        }





if __name__ == "__main__":
    # Create Reddit client
    reddit_client = RedditClient()

    # Specify subreddit
    subreddit = 'python'

    # Get latest posts
    posts = reddit_client.get_subreddit_posts(subreddit, limit=5)
    print(posts)