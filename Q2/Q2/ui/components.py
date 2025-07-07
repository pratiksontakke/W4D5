"""UI components for the Legal Document Search System."""


class SearchInterface:
    """
    Search interface component for handling user queries and displaying results.
    """

    def __init__(self, search_client):
        """
        Initialize search interface.

        Args:
            search_client: Client for performing searches
        """
        self.search_client = search_client
