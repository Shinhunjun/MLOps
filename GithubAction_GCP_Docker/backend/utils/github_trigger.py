"""
GitHub Actions trigger via Repository Dispatch
"""
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GitHubActionsTrigger:
    """Trigger GitHub Actions workflows via Repository Dispatch"""

    def __init__(self, github_token: str, owner: str, repo: str):
        """
        Initialize GitHub Actions trigger

        Args:
            github_token: GitHub Personal Access Token
            owner: Repository owner (e.g., "Shinhunjun")
            repo: Repository name (e.g., "MLOps")
        """
        self.github_token = github_token
        self.owner = owner
        self.repo = repo
        self.api_url = f"https://api.github.com/repos/{owner}/{repo}/dispatches"

    def trigger_retrain(
        self,
        data_count: int,
        sub_set_id: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Trigger model retraining workflow

        Args:
            data_count: Number of new data samples
            sub_set_id: Subset identifier (e.g., "sub_set_5")
            additional_data: Additional data to pass to the workflow

        Returns:
            True if trigger was successful, False otherwise
        """
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        payload = {
            "event_type": "retrain-model",
            "client_payload": {
                "data_count": data_count,
                "dataset_version": sub_set_id,
                "trigger_source": "fastapi_feedback",
                **(additional_data or {})
            }
        }

        try:
            logger.info(f"Triggering GitHub Actions workflow: {self.api_url}")
            logger.info(f"Payload: {payload}")

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 204:
                logger.info("✅ GitHub Actions workflow triggered successfully!")
                return True
            else:
                logger.error(f"❌ Failed to trigger workflow: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Error triggering GitHub Actions: {e}")
            return False
