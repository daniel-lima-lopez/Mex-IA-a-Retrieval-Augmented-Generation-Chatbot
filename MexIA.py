# se cargan las librerias necesarias
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os

class MexIA_ChatBot:
    def __init__(self, temperature=0.5, model='gpt-4o-mini', api_path='openai_key.txt'):
        # lectura del api de open ai
        key = ''
        with open(api_path, 'r') as t:
            key = t.read()
        os.environ["OPENAI_API_KEY"] = key

        # configuracion del modelo y embeding
        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.embed_model = embed_model
        Settings.llm = OpenAI(model=model, temperature=temperature)

        # inicia el cliente
        db = chromadb.PersistentClient(path="./index_data")
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # lee el index del contenido almacenado
        self.index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

        # definicion del Retiever considerando los 5 nodos mas similares
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,
        )
        
        #definicion del motor de busqueda
        self.query_engine = self.index.as_query_engine()


    # implementacion del diseno de promt y su respuesta dada una consulta
    def get_answer(self, query):        
        # contexto de la respuesta
        context = "Eres un asistente legal entrenado para proporcionar información sobre la "\
        "Constitución de México. Tu tarea es responder preguntas legales específicas consultando "\
        "la Constitución y proporcionar la respuesta exacta. "

        # extraccion de informacion
        nodes = self.retriever.retrieve(query)
        
        # accede a la informacion del retrieve dado el query
        docs = ''
        for ni in nodes:
            aux_dic = dict(ni)
            #print(aux_dic)
            docs += f"DOCUMENT NAME: Document {aux_dic['node'].metadata['nombre']}  CONTENT: {aux_dic['node'].text} \n"

        ret = "Asegúrate de citar el o los artículos correspondientes de donde sacaste la "\
        f"información utilizando los siguientes documentos {docs}. "

        # few shot learning y definicion del formato de respuesta
        format = "Sé conciso pero preciso en tus respuestas. Si la pregunta se refiere a derechos "\
        "humanos, menciona la sección de la Constitución que habla sobre derechos fundamentales. "\
        "Si la pregunta es sobre procedimientos legales o derechos específicos, dirígete al artículo "\
        "relevante que trata ese tema. Formato de respuesta: Respuesta: [tu respuesta clara y concisa] "\
        "Fuente: Artículo [número del artículo] Ejemplo: Pregunta: ¿Qué derechos tienen los ciudadanos "\
        "mexicanos en cuanto a la libertad de expresión? Respuesta: Los ciudadanos mexicanos tienen "\
        "derecho a expresarse libremente, sin interferencias, siempre y cuando no se atente contra la "\
        "moral, los derechos de terceros, o provoque algún delito o disturbio. Fuente: Artículo 6 de "\
        "la Constitución de México. "

        # pregunta del usuario
        aux_query = f"La pregunta es {query}"

        # promt final
        promt = context + ret + format + aux_query

        # generacion de respuesta
        response = self.query_engine.query(promt)
        
        return response.response