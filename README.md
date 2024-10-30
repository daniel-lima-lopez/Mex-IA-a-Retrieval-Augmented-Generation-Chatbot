# MexIA: chatbot basado en RAG
En este repositorio se muestra el proceso de procesamiento de información y la implementación del sistema MexIA. El objetivo de este proyecto es diseñar un chatbot basado en RAG (Retrieval Augmented Generation) para contestar consultas de carácter legal, considerando la información contenida en la constitución Mexicana.

El reporte completo de este proyecto se encuentra en el archivo [Reporte MexIA.pdf](Reporte MexIA.pdf).


## Instalación
Para instalar este proyecto primero es necesario clonar este repositorio:
```bash
git clone git@github.com:daniel-lima-lopez/Mex-IA-a-Retrieval-Augmented-Generation-Chatbot.git
```
También es necesario instalar Llama Index ([ver mas detalles](https://docs.llamaindex.ai/en/stable/getting_started/installation/)). Cabe mencionar que este proyecto emplea el modelo de lenguaje pre entrenado GPT-4o-mini, por lo cual es necesario contar con una key del API de OpenAI ([tutorial](https://medium.com/@lorenzozar/how-to-get-your-own-openai-api-key-f4d44e60c327)).

Una vez obtenida la key, es necesario crear un archivo nombrado `openai_key.text` y guardar la key en el.

## Implementación del modelo Mex-IA
La implementación del modelo considera los siguientes puntos:
1. Definición del modelo de lenguaje y tipo de embedding a usar.
2. Recuperación de la representación vectorial de la información de los artículos de la constitución Mexicana.
3. Definición del motor de recuperación de información y el modelo de consultas.
4. Implementación de consultas considerando el siguiente esquema para personalizar el prompt evaluado por el modelo de lenguaje:

![png](Diagrama_RAG.png)

La implementación del modelo propuesta se encuentra en el archivo [MexIA.py](MexIA.py). A continuación se muestra dicha implementación:

```python
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
```

El  modelo puede instanciarse importando el archivo, y posteriormente definiendo una instancia del ChatBot:


```python
# se importa el modelo implementado
from MexIA import MexIA_ChatBot
# se instancia al chatbot
chat = MexIA_ChatBot()
```

A continuación se muestran 15 ejemplos usados para evaluar el desempeño del modelo:

```python
querys = ["¿Cual es el conocimiento basico que debo tener como mexicano?", "¿Cuales son mis derechos como trabajador?",
          "¿En la constitucion, que leyes cuidan la naturaleza?", "¿Como esta compuesta la constitucion, que temas abarca?",
          "¿Bajo que circunstancias me puede detener la policia?", "¿La constitucion ampara mi derecho a la educacion?",
          "¿Cuál es la diferencia entre un delito y una falta cívica?","¿Que es un delito doloso?",
          "¿Que requisitos debe cumplir una pesona para obtener la nacionalidad mexicana?",
          "¿Que impuestos debo pagar por tener una propiedad?", "¿Que pasaria si no pago impuestos?",
          "¿Que articulo de la constitucion respalda mi libertad de expresion?",
          "¿Cuales son los derechos y obligaciones de mis padres?","¿Que documentos necesito para registar a mi bebe?",
          "¿Los migrantes tienen derecho a votar?"]

for qi in querys:
    print(f'CONSULTA: {qi}')
    anws = chat.get_answer(qi)
    print(f'{anws}\n')
```
    CONSULTA: ¿Cual es el conocimiento basico que debo tener como mexicano?
    Respuesta: Como mexicano, es fundamental conocer tus obligaciones, que incluyen asegurar que tus hijos reciban educación, participar en la instrucción cívica y militar, alistarte en los cuerpos de reserva, y contribuir a los gastos públicos de manera proporcional. Además, es importante entender tus derechos humanos, que son garantizados y protegidos por la Constitución y los tratados internacionales.
    
    Fuente: Artículo 31 y Artículo 1o.
    
    CONSULTA: ¿Cuales son mis derechos como trabajador?
    Como trabajador, tienes derechos fundamentales que incluyen, entre otros:
    1. Derecho a un trabajo digno y socialmente útil.
    2. Jornada máxima de trabajo de ocho horas diarias y siete horas nocturnas.
    3.  Derecho a un día de descanso por cada seis días de trabajo.
    4.  Vacaciones de al menos veinte días al año.
    5.  Salario mínimo que debe ser suficiente para satisfacer las necesidades normales de una familia.
    6.  Igualdad de salario por trabajo igual, sin distinción de sexo.
    7.  Derecho a la capacitación y adiestramiento.
    8.  Protección contra despidos injustificados, con derecho a reinstalación o indemnización.
    9.  Derecho a organizarse en sindicatos y a la huelga.
    Fuente: Artículo 123 de la Constitución de México.
    
    CONSULTA: ¿En la constitucion, que leyes cuidan la naturaleza?
    Las leyes que cuidan la naturaleza están relacionadas con la propiedad de las aguas, los recursos naturales y la protección de tierras. La Constitución establece que la propiedad de la Nación incluye las aguas de los mares territoriales, lagos, ríos y sus afluentes, así como la regulación de su uso y explotación. También se reconoce la propiedad de la tierra por parte de los núcleos de población ejidales y comunales, protegiendo su integridad para actividades productivas y el asentamiento humano.
    
    Fuente: Artículo 27 y Artículo 73 de la Constitución de México.
    
    CONSULTA: ¿Como esta compuesta la constitucion, que temas abarca?
    La Constitución está compuesta por varios artículos que abarcan una amplia gama de temas, incluyendo derechos fundamentales, la organización del Estado, la división de poderes, la regulación de los derechos de autor, la contabilidad gubernamental, y la resolución de controversias constitucionales. También incluye disposiciones sobre la legislación en materia de seguridad, educación, y la participación de las entidades federativas en diversos asuntos. 
    
    Fuente: Artículo 73, Artículo 105.
    
    CONSULTA: ¿Bajo que circunstancias me puede detener la policia?
    La policía puede detener a una persona en el momento en que esté cometiendo un delito o inmediatamente después de haberlo cometido, debiendo ponerla sin demora a disposición de la autoridad civil más cercana. También se puede ordenar la detención en casos urgentes, cuando se trate de un delito grave y exista el riesgo de que la persona se sustraiga a la acción de la justicia, siempre que no se pueda acudir ante la autoridad judicial por razones de hora, lugar o circunstancia. 
    
    Fuente: Artículo 16.
    
    CONSULTA: ¿La constitucion ampara mi derecho a la educacion?
    El derecho a la educación está garantizado para toda persona, y el Estado tiene la responsabilidad de impartir y garantizar diversos niveles educativos, incluyendo la educación básica, que es obligatoria. Además, se establece que la educación debe ser universal, inclusiva, pública, gratuita y laica. 
    
    Fuente: Artículo 3o de la Constitución de México.
    
    CONSULTA: ¿Cuál es la diferencia entre un delito y una falta cívica?
    Los delitos son conductas tipificadas en la legislación penal que implican una sanción penal, que puede incluir penas privativas de libertad o multas, y que afectan a la sociedad en su conjunto. Por otro lado, las faltas cívicas son infracciones menores que se sancionan administrativamente y no implican necesariamente una pena privativa de libertad. Estas últimas pueden resultar en amonestaciones, multas o sanciones administrativas. 
    
    Fuente: Artículo 109 y Artículo 111 de la Constitución de México.
    
    CONSULTA: ¿Que es un delito doloso?
    Un delito doloso es aquel en el que el autor actúa con intención de cometer el delito, es decir, tiene la voluntad de realizar la conducta delictiva y de causar el resultado que se tipifica como delito en la ley.
    
    Fuente: Artículo 14 de la Constitución de México.
    
    CONSULTA: ¿Que requisitos debe cumplir una pesona para obtener la nacionalidad mexicana?
    Respuesta: Para obtener la nacionalidad mexicana por naturalización, una persona debe cumplir con los siguientes requisitos: ser extranjero y obtener de la Secretaría de Relaciones una carta de naturalización, o ser extranjero que contraiga matrimonio con un mexicano y establezca su domicilio en el territorio nacional, cumpliendo con los demás requisitos que la ley señale.
    
    Fuente: Artículo 30.
    
    CONSULTA: ¿Que impuestos debo pagar por tener una propiedad?
    Los impuestos que debes pagar por tener una propiedad incluyen las contribuciones que establezcan los Estados sobre la propiedad inmobiliaria, así como las tasas adicionales que puedan aplicarse por su fraccionamiento, división, consolidación, traslación y mejora. Si no pagas estos impuestos, podrías enfrentar sanciones que van desde multas hasta la posibilidad de que se inicie un procedimiento de embargo sobre la propiedad.
    
    Fuente: Artículo 115, incisos a) y b).
    
    CONSULTA: ¿Que pasaria si no pago impuestos?
    Respuesta: Si no se realizan los pagos de impuestos, se pueden generar sanciones administrativas, que incluyen multas y recargos. Además, la falta de pago puede dar lugar a procedimientos legales que pueden culminar en acciones de cobro por parte de las autoridades fiscales.
    
    Fuente: Artículo 109.
    
    CONSULTA: ¿Que articulo de la constitucion respalda mi libertad de expresion?
    La libertad de expresión está respaldada por el derecho a manifestar ideas sin ser objeto de inquisición, salvo en casos que atenten contra la moral, la vida privada o los derechos de terceros, así como en situaciones que provoquen delitos o perturben el orden público
    
    Fuente: Artículo 6 de la Constitución de México.
    
    CONSULTA: ¿Cuales son los derechos y obligaciones de mis padres?
    Respuesta: Los padres tienen la obligación de preservar y exigir el cumplimiento de los derechos de sus hijos, así como de asegurar que sus hijos concurran a las escuelas para recibir la educación obligatoria. Además, tienen el derecho de participar en el proceso educativo de sus hijos, velando por su bienestar y desarrollo.
    
    Fuente: Artículo 4o y Artículo 31 de la Constitución de México.
    
    CONSULTA: ¿Que documentos necesito para registar a mi bebe?
    Respuesta: Para registrar a tu bebé, necesitarás la primera copia certificada del acta de registro de nacimiento, que debe ser expedida gratuitamente por la autoridad competente.
    
    Fuente: Artículo 4o.
    
    CONSULTA: ¿Los migrantes tienen derecho a votar?
    Respuesta: Los migrantes no tienen derecho a votar en las elecciones populares en México, ya que este derecho está reservado para los ciudadanos mexicanos. Los derechos de la ciudadanía, que incluyen el derecho a votar, están específicamente establecidos en el artículo 35 de la Constitución.
    
    Fuente: Artículo 35.