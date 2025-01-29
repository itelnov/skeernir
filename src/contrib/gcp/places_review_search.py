import requests


class GoogleReviewSearcher:
    """
    The object responsible for handling Google Places API requests within
    langchain pipeline
    """
    BASE_URL = (
        "https://maps.googleapis.com/maps/api/place/"
        "findplacefromtext/json"
        )
    DETAILS_URL = (
        "https://maps.googleapis.com/maps/api/place/"
        "details/json"
        )

    def __init__(self, api_key: str):
        self._api_key = api_key

    def __call__(
            self, location_desc: str, place_id: str = None) -> dict | None:

        if place_id is None:
            try:
                place_id = self._get_place_id(location_desc)
            except BaseException:
                print('Could not index the queried location')
                return

        params = {
            'place_id': place_id,
            'key': self._api_key,
            'fields': 'reviews'
        }

        response = requests.get(self.DETAILS_URL, params=params)
        result = response.json()
        return result.get('result', {}).get('reviews')

    def _get_place_id(self, location_desc: str):
        params = {
            'input': location_desc,
            'inputtype': 'textquery',
            'fields': 'place_id',
            'key': self._api_key
        }

        response = requests.get(self.BASE_URL, params=params)
        result = response.json()
        return result['candidates'][0]['place_id']


if __name__ == "__main__":
    places_api_key = ""
    review_searcher = GoogleReviewSearcher(places_api_key)

    place_ids = {
        "ChIJLU7jZClu5kcR4PcOOO6p3I0": "Eifel Tower France"
    }

    reviews = review_searcher('Eifel Tower France')
    for review in reviews:
        print(review['text'])
