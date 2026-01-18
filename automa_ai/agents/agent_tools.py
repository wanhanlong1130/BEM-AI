from langchain_core.tools import tool


@tool
def load_skill(skill_name: str) -> str:
    """
        Load a specialized skill prompt. as defined by user
        Sample available skill prompt from user prompt would be:

        `
            Available skills:
            - write_sql: SQL query writing expert
            - review_legal_doc: Legal document reviewer
        `

        Returns the skill's prompt and context.
    """
    # Load skill content from file/database

