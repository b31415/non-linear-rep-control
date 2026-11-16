"""
Helpers used by representation methods
"""
import re
import copy
from typing import Callable

def apply_chat_template_for_incomplete_response(chat_template: list[dict], tokenizer) -> str:
    """
    Converts the chat template to a str (similarly to tokenizer.apply_chat_template),
    except that the partial response is appended to the formatted response
    such that the </s> tag is not added.
    
    Args:
        chat_template: Huggingface chat template where the last dict has "assitant" role
        tokenizer
    """
    if chat_template[-1]["role"] != "assistant":
        raise ValueError('Expected last role to be "assistant"')
    
    prompt, response = chat_template[:-1], chat_template[-1]
    if len(prompt) == 0:
        prompt_str = tokenizer.bos_token
    else:
        prompt_str = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True, 
    )
    prompt_str += response["content"]
    return prompt_str

def apply_chat_template(chat_template: list[dict], tokenizer) -> str:
    """
    Format chat_template in the same way as apply_concept_template(),
    but without prepending instructions to act with a concept.
    """
    chat_template = chat_template.copy()
    if chat_template[-1]["role"] != "assistant":
        chat_template.append({"role": "assistant", "content": ""})

    return apply_chat_template_for_incomplete_response(chat_template, tokenizer)

def apply_concept_template(
        prompt: list[dict],
        tokenizer,
        prompt_templates: dict[float, str | dict | None],
        concept_words: tuple = ('honest', 'dishonest', 'tell the truth'),
        process_prompt: Callable[[str, float], str] = lambda prompt, _: prompt,
        prepend_concept_instruction: bool = True
    ) -> tuple[list, list]:
        """
        Args:
            prompt: Prompt in the form of a Huggingface chat template to
                be converted to a str.
            prompt_templates:
                Keys are concept strengths
                Values are optional instructions to prepend to prompt
            concept_words: A tuple of words corresponding to values to fill
                in the blanks for <prompt_templates>. E.g.
                (concept adjective, antonym, description)
            process_prompt: Function in the form func(prompt, control_strength)
                Supply this if the prompt should be modified given a concept strength.
                Only modifies the first content in the chat template.
            prepend_concept_instruction: Whether to prepend or append the concept instruction to the prompt.
        Returns:
            (train_inputs, train_labels): Values for passing into get_directions() to get the
            representation for the current prompt.
        """
        def concat_instruction(instruction: str, prompt: str):
            return f"{instruction} {prompt}" if prepend_concept_instruction else f"{prompt} {instruction}"
        def format_prompt(chat_template: list[dict], control_strength: float, template: str | dict | None) -> str:
            """
            Convert chat template to a str with an instruction to act with a concept
            prepended/appended, modifying the prompt in relation to the concept if required.
            """
            chat_template = process_prompt(chat_template, control_strength)
            
            # Prepend/append instruction to prompt
            if template is not None:
                instruction = template.format(concept_words=concept_words)
                if isinstance(chat_template, str):
                    # chat_template is already a str prompt
                    return concat_instruction(instruction, chat_template)
                
                elif chat_template[0]["role"] == "assistant":
                    # If there is no prompt/user, make the instruction the prompt
                    chat_template.insert(0, {"role": "user", "content": instruction})
                else:
                    chat_template[0]["content"] = concat_instruction(instruction, chat_template[0]['content'])
            
            # Convert chat template to str
            return apply_chat_template(chat_template, tokenizer)

        formatted_prompts = [
            format_prompt(copy.deepcopy(prompt), control_strength, template) \
            for control_strength, template in prompt_templates.items()
        ]
        labels = list(prompt_templates.keys())
        return formatted_prompts, labels

def get_possible_answers(question: str):
    """
    Parse question to get valid options from a multiple choice question.
    """
    pattern = re.compile('.\. ')
    question = question.split('\n')
    possible_answers = [line[0] for line in question if pattern.match(line)]
    return possible_answers