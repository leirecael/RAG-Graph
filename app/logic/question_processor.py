from llm.llm_client import call_llm
import json

async def determine_simplicity(question: str) -> bool:
    prompt = f"""
        Tu tarea es determinar si la siguiente pregunta del usuario es **simple** o **compleja** en el contexto de generar consultas Cypher.

        ### Criterios:
        - Una **pregunta simple** es aquella que puede ser respondida con una única instancia de cada tipo de entidad. Por ejemplo, un solo problema, un solo contexto, un solo stakeholder, etc. Se permiten múltiples tipos de entidades, siempre que no se repita el mismo tipo.
        - Una **pregunta compleja** es aquella que requiere usar **más de una instancia del mismo tipo de entidad** en la consulta Cypher, por ejemplo dos problemas relacionados con el mismo contexto o diferentes stakeholders, etc.

        ### Ejemplos:
        Pregunta: ¿Qué problemas tienen los desarrolladores de software?
        Respuesta: {{"is_simple": true}}

        Pregunta: ¿Qué problemas afectan al mismo contexto?
        Respuesta: {{"is_simple": false}}

        Pregunta: ¿Qué problemas afectan al mismo contexto pero diferentes stakeholders?
        Respuesta: {{"is_simple": false}}

        Pregunta: ¿Qué objetivos están relacionados con el requisito R1?
        Respuesta: {{"is_simple": true}}

        Pregunta: ¿Qué requisitos están asociados a problemas distintos dentro del mismo contexto?
        Respuesta: {{"is_simple": false}}

        Devuelve únicamente la respuesta en formato JSON como:
        {{"is_simple": true}}

        Pregunta: {question}
    """
    response, cost = await call_llm(prompt,model="gpt-3.5-turbo")
    simplicity = json.loads(response)   
    return simplicity, cost

async def extract_entities(question: str) -> dict:
    prompt = f"""
        Extrae las siguientes entidades de la pregunta que se muestra a continuación:
        - problem: A problem represents a deficiency, shortfall, or issue that the artifact aims to resolve. Keywords are usually negative, like lack of, etc.
        - stakeholder: A stakeholder is any individual, group, or entity invested in or affected by the artifact’s development or performance.
        - goal: A goal represents the desired high-level outcome that stakeholders seek to achieve by using the artifact. Goals reflect aspirations for improvements, efficiencies, or enhancements within a given domain. Uses keywords like enchance, improve, etc.
        - context: Context refers to the specific environment, conditions, or situational factors where the artifact will be used.
        - requirement: A requirement defines the specific functionalities, constraints, or conditions that the artifact must fulfill to achieve the stated goal.
        - artifactClass: ArtifactClass represents a general category or class of solutions proposed to address the problem. It describes the functional principles of the artifact rather than a specific tool or implementation.

        Relación entre tipos de nodos:
        - (problem)-[:arisesAt]->(context)
        - (problem)-[:concerns]->(stakeholder)
        - (problem)-[:informs]->(goal)
        - (requirement)-[:meetBy]->(artifactClass)
        - (problem)-[:addressedBy]->(artifactClass)
        - (goal)-[:achievedBy]->(requirement)

        Intenta que el valor de las entidades sea los más completo posible.
        El usuario suele querer respuestas de un tipo, por ejemplo soluciones a un problema, problemas que ocurren en cierto contexto, requerimientos de cierto objetivo, etc..
        Devuelve el resultado en formato JSON. Si una entidad no está presente, no la pongas. Para lo que desea el usuario saber, indicalo con PrimaryQuestion.
        Es posible que la pregunta solo contenga un tipo de entidad y ese tipo de entidad sea al mismo tiempo el PrimaryQuestion. 
        En este caso pon la entidad dos veces, una con su valor y la otra con PrimaryQuestion. 
        Puede haber varios PrimaryQuestion de diferentes entidades, pero para cada entidad solo puede haber un PrimaryQuestion.
        Si no encuentras una entidad explícitamente mencionada o inferida claramente en la pregunta, no la inventes ni rellenes.


        Ejemplo: 
        ***Pregunta: ¿Cómo podría resolver el problema de trazabilidad en entornos hospitalarios usando sensores IoT?
        {{
        "problem": "trazabilidad en entornos hospitalarios",
        "stakeholder": "hospitales",
        "context": "sensores IoT",
        "artifactClass": "PrimaryQuestion",
        }}***
        ***Pregunta: ¿Qué problemas tienen los desarrolladores de software?
        {{
        "stakeholder": "desarrolladores de software",
        "problem": "PrimaryQuestion"
        }}***
        ***Pregunta: ¿Qué son los desarrolladores de software?
        {{
        "stakeholder": "desarrolladores de software",
        "stakeholder": "PrimaryQuestion"
        }}***
        ***Pregunta: ¿Quienes son afectados por el cambio climatico y como podemos solucionarlo?
        {{
        "problem": "el cambio climatico",
        "stakeholder": "PrimaryQuestion"
        "artifactClass": "PrimaryQuestion"
        }}***
        ***Pregunta: ¿Qué problemas afectan al mismo contexto?
        {{
        "problem": "PrimaryQuestion"
        "context": "PrimaryQuestion"
        }}***
        **Pregunta: ¿Qué problemas afectan al mismo contexto pero diferentes stakeholders?**
        {{
            "problem": "PrimaryQuestion",
            "context": "PrimaryQuestion",
            "stakeholder": "PrimaryQuestion"
        }}***
        
        Pregunta:
        {question}
    """
    response, cost = await call_llm(prompt)
    entities = json.loads(response)
    

    return entities, cost