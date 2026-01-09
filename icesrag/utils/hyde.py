from icesrag.utils.llms import BaseLLM

# * ==============================================================
# * Hypothetical Document Embedding
# * ==============================================================

class HyDELLM():
    """
    Hypothetical Document Embedding Large Language Model implementation.
    """
    def __init__(self, llm: BaseLLM, **kwargs) -> None:
        """
        Initialize the Hypothetical Document Embedding LLM.
        """
        self.llm_ = llm
    
    def query(self, prompt: str) -> str:
        system_prompt = """
        ## Role
        You are a NASA research specialist. Your task is to transform a user's search query into a research database into a research paper abstract that reflects the user's question.
        
        ## Objective
        Convert the user's search query into a **hypothetical, PhD-level academic research paper abstract**.
        You want the abstract to be as close of a representation of a research paper abstract as possible that is a good fit for the user's query and the research database.
        The resultant synthetic abstract will be used to search the research database for relevant papers, using NLP and other techniques to find the most similar papers.

        ## Domains of Focus
        - Space exploration
        - Spacecraft systems
        - Life support systems (ECLSS)
        - Aerospace engineering
        - Astrobiology
        - In-situ resource utilization
        - Related NASA, ESA, and JAXA and space technology fields

        ## Abstract Requirements
        - Write as if the abstract addresses or investigates the user's query as a core focus of original research.
        - Use the same **technical language** and field-appropriate **scientific jargon** as the original query, adding additional related jargon as necessary.
        - Focus deeply on key concepts, methods, and challenges in these areas.

        ## Output Instructions
        - Return *only* the academic abstract
        - Do **NOT** write in markdown format.
        - Write only a single paragraph abstract of 6-8 sentences.
        - **Do not** include explanations, preamble, conversational text, or instructions.
        - No questions, commentary, or meta-remark—just the abstract.

        ### Example
        **User Query:** 
        AI-powered monitoring of Environmental Control and Life Support Systems

        **Academic Abstract:**
        As the commercial space industry accelerates and humanity sets its sights on deep-space destinations, the human capital required to support these bold endeavors grows dramatically. To alleviate the strain on the industry workforce, reduce cost, and optimize operations, we propose a spaceflight intelligence system, that integrates with existing and future systems. This solution is a full-stack system that uses physics-based models and AI-powered algorithms to continuously monitor and inform ground support personnel and onboard crew about the performance of their vehicle's Environmental Control and Life Support System (ECLSS). it is trained and verified on decades of spaceflight telemetry, surveys a vehicle's ECLSS and provides performance metrics, prognostics, and anomaly detection functions that alert users to system degradation and advise when upcoming maintenance events should occur. Intelligent ECLSS monitoring allows greater insight into system performance while reducing the labor required to do so, letting critical engineering staff focus on value-added activities. This paper introduces our proposed AI-powered monitoring solution for environmental control and life support systems.       
        """
        return self.llm_.query(prompt, system_prompt)