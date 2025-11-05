import os
import requests
from typing import List
from ...config.model_config import ImageSearchConfig

class ImageSearch:
    def __init__(self, config: ImageSearchConfig):
        self.config = config
        self.api_key = config.api_key
        self.search_engine_id = getattr(config, 'search_engine_id', '')
        
    def search(self, query: str, num: int = 10) -> List[str]:
        """
        Search for images using Google Custom Search API
        
        Args:
            query: Search query
            num: Number of results to return (max 10)
            
        Returns:
            List of image URLs
        """
        try:
            # Google Custom Search API endpoint
            url = "https://www.googleapis.com/customsearch/v1"
            
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'searchType': 'image',
                'num': min(num, 10),  # Google API max is 10
                'safe': 'high'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract image URLs
            image_urls = []
            for item in data.get('items', []):
                image_url = item.get('link')
                if image_url:
                    image_urls.append(image_url)
            
            return image_urls
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            return []
        except Exception as e:
            print(f"Error processing search results: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if the service is available"""
        return bool(self.api_key and self.search_engine_id)
