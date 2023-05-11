import tiktoken, openai, os

from typing import List
from openai.embeddings_utils import get_embedding

class ChatGPT:

    def __init__(self, engine: str = 'gpt-3.5-turbo', embedding_engine:str = "text-embedding-ada-002"):
        """
        Class that encapsulates the functions to interacting with the OpenAI API.
        
        Attribuites:
        ------------
        engine: str
            OpenAI engine version.
        
        Raises:
        ------------
        No raises.
        """
        
        openai.api_key: str = os.getenv('OPENAI_CODECHAT_KEY')
        self.engine = engine
        self.embedding_engine = embedding_engine
    
    def models(self) -> List[str]:
        """
        Returns a list of avalilable models.
        
        Parameters:
        ------------
        No parameters.
        
        Raises:
        ------------
        No raises.
        """
        
        models: List[str] = openai.Model.list()
        return [m['id'] for m in models['data']]
    
    def chat(self, prompt: str) -> str:
        """
        Performs completion inference.
        
        Parameters:
        ------------
        prompt: str
            Text to completion.
        
        Raises:
        ------------
        No raises.
        """
        
        resp: openai.openai_object.OpenAIObject = openai.ChatCompletion.create(
          model=self.engine,
          presence_penalty=1,
          temperature=0,            
          messages=[
              {'role': 'user', 'content': prompt}
          ]
        )
        
        choices: List[str] = resp.choices
        
        return None if len(choices) == 0 else choices[0].message.content

    def embeddings(self, text, max_tokens=4096):
        """
        Returns the embedding array for sentence.

        Parameters:
        ------------
        text: str
            Text to vectorize.

        max_tokens: int
            Max tokens

        Raises:
        ------------
        No raises.
        """

        if self.count_tokens(text) > max_tokens:
            text = self.trim_tokens(text, max_tokens)

        return get_embedding(text, engine=self.embedding_engine)
    
    def count_tokens(self, s: str) -> int:
        """
        Returns the number of tokens in a text string.

        Parameters:
        ------------
        s: str
            Sentence to count.

        max_tokens: int
            Max tokens

        Raises:
        ------------
        No raises.
        """

        encoding = tiktoken.encoding_for_model(self.engine)
        return len(encoding.encode(s))
    
    def trim_tokens(self, s: str, n: int) -> int:
        """
        Returns a fragment of embedding.

        Parameters:
        ------------
        s: str
            Sentence to embedding.

        n: int
            Trim position

        max_tokens: int
            Max tokens

        Raises:
        ------------
        No raises.
        """

        encoding = tiktoken.encoding_for_model(self.engine)
        tokens = encoding.encode(s)[:n]
        return encoding.decode(tokens)

    
