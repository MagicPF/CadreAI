import re
import json
import os

from accelerate.commands.config.config_args import cache_dir
from gradio_client import Client


class ProteinExpert:
    def __init__(self, server_url="http://search-protrek.com/", cache_file="protein_cache.json",cache_dir="./jsons/"):
        """
        Initializes the ProteinExpert with local caching.

        Args:
            server_url (str): URL of the external protein feature retrieval service.
            cache_file (str): Path to the local cache file.
        """
        self.client = None
        self.server_url = server_url
        self.cache_file = cache_dir+cache_file
        self.cache = self._load_cache()

        self.interaction_subsections = [
            "Activity regulation", "Catalytic activity", "Cofactor", "Domain (non-positional annotation)",
            "Enzyme commission number", "Function", "GO annotation", "Induction", "Pathway",
            "Post-translational modification", "Sequence similarities", "Subcellular location", "Subunit"
        ]
        self.sideeffect_subsections = [
            "Allergenic properties", "Biophysicochemical properties", "Caution", "Disruption phenotype",
            "Involvement in disease", "Miscellaneous", "Pharmaceutical use", "Polymorphism", "Tissue specificity",
            "Toxic dose"
        ]

    def _load_cache(self):
        """Loads the cache from a file if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Saves the cache to a file."""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=4)

    def retrieval(self, sequence: str, subsection_type: str = "Function", nprobe: int = 1000, topk: int = 1, db: str = "Swiss-Prot") -> str:
        """
        Retrieves protein-related data from a database. Uses caching to avoid redundant queries.

        Args:
            sequence (str): Protein sequence in FASTA format.
            subsection_type (str): The type of information to retrieve.
            nprobe (int): Number of nearest neighbors to search.
            topk (int): Number of top results to return.
            db (str): The database to query.

        Returns:
            str: Retrieved protein feature description.
        """
        if subsection_type not in self.interaction_subsections + self.sideeffect_subsections:
            raise ValueError(f"Invalid subsection_type: {subsection_type}. Choose from: {self.interaction_subsections + self.sideeffect_subsections}")

        cache_key = f"{sequence}_{subsection_type}_{db}"
        if cache_key in self.cache.keys():
            return self.cache[cache_key]  # Return cached result

        try:
            if self.client is None:
                self.client = Client(self.server_url)
            result = self.client.predict(
                input=sequence,
                nprobe=nprobe,
                topk=topk,
                input_type="sequence",
                query_type="text",
                subsection_type=subsection_type,
                db=db,
                api_name="/search"
            )

            text = result[0]  # Assuming result is always in this format
            match = re.search(r"\|  0 \| (.*?) \|", text)
            response = match.group(1).strip() if match else "Description not found."

            # Save result to cache
            self.cache[cache_key] = response
            self._save_cache()

            return response

        except Exception as e:
            return f"Error occurred: {str(e)}"

    def retrieval_all(self, sequence: str, nprobe: int = 1000, topk: int = 1, db: str = "Swiss-Prot") -> dict:
        """
        Retrieves all available protein-related data.

        Args:
            sequence (str): Protein sequence in FASTA format.
            nprobe (int): Number of nearest neighbors to search.
            topk (int): Number of top results to return.
            db (str): The database to query.

        Returns:
            dict: A dictionary with retrieved protein features.
        """
        results = {}
        for subsection in self.interaction_subsections + self.sideeffect_subsections:
            results[subsection] = self.retrieval(sequence, subsection, nprobe, topk, db)
        return results

    def retrieval_interaction(self, sequence: str, nprobe: int = 1000, topk: int = 1, db: str = "Swiss-Prot") -> dict:
        """
        Retrieves protein interaction-related features.

        Args:
            sequence (str): Protein sequence in FASTA format.
            nprobe (int): Number of nearest neighbors to search.
            topk (int): Number of top results to return.
            db (str): The database to query.

        Returns:
            dict: A dictionary with retrieved protein interaction features.
        """
        results = {}
        for subsection in self.interaction_subsections:
            results[subsection] = self.retrieval(sequence, subsection, nprobe, topk, db)
        return results

    def retrieval_sideeffect(self, sequence: str, nprobe: int = 1000, topk: int = 1, db: str = "Swiss-Prot") -> dict:
        """
        Retrieves protein side-effect-related features.

        Args:
            sequence (str): Protein sequence in FASTA format.
            nprobe (int): Number of nearest neighbors to search.
            topk (int): Number of top results to return.
            db (str): The database to query.

        Returns:
            dict: A dictionary with retrieved protein side-effect features.
        """
        results = {}
        for subsection in self.sideeffect_subsections:
            results[subsection] = self.retrieval(sequence, subsection, nprobe, topk, db)
        return results


# Example usage
if __name__ == "__main__":
    protein_expert = ProteinExpert()

    sequence = "MSATAEQNARNPKGKGGFARTVSQRPDLFREGQGVVAEGRFGSDGLFRADNVLAKHDENYVPKDLADSLKKKGVWEGK"

    print("🔍 Retrieving protein interaction information...")
    interaction_results = protein_expert.retrieval_interaction(sequence)
    print("Interaction Info:", json.dumps(interaction_results, indent=4))

    print("\n⚠️ Retrieving protein side effects information...")
    sideeffect_results = protein_expert.retrieval_sideeffect(sequence)
    print("Side Effect Info:", json.dumps(sideeffect_results, indent=4))