def get_dataset_prompts(dataset):    
    # Dataset-specific conditioning prompts (only to be used in instriction+Q/instruction+few-shot+Q setting)
    if dataset == "COMPS":
        prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Does a blueberry fire bullets?\nResponse: No\n\n#EXAMPLE\nQuestion: Does a turtle have a hard shell?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "

        prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "

    elif dataset == "EWOK":
        prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Claire sees something that is fabric. Can Claire pour it?\nResponse: No\n\n#EXAMPLE\nQuestion: Sally pays salary to Harry. Is Sally Harry's boss?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "

        prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "

    elif dataset == "BABI":
        prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Marshall is in the car. Is Marshall in the building?\nResponse: No\n\n#EXAMPLE\nQuestion: Nathan is a pianist. Pianists like oranges. Does Nathan like oranges?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "

        prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "

    elif dataset == "ARITH":
        prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Is 7 minus 9 equal to 4?\nResponse: No\n\n#EXAMPLE\nQuestion: Is 17 plus 15 equal to 32?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "

        prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "

    elif dataset == "MMLU":
        prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following multiple choice questions:\n\n#EXAMPLE\nQuestion: What is the shape of the Earth?\nOptions: (A) Cone, (B) Cube, (C) Sphere, (D) Cylinder\nResponse: C\n\n#EXAMPLE\nQuestion: What is the color of the sky?\nOptions: (A) Red, (B) Blue, (C) Green, (D) Yellow\nResponse: B\n\n#EXAMPLE\nQuestion: "

        prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following multiple choice question:\n\n#EXAMPLE\nQuestion: "

    return prompt_cond1, prompt_cond2