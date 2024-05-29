

class BaseFormatter:
    def __init__(self, task_description: str, instructions: str):
        self.task_description = task_description
        self.instructions = instructions

    def format_prompt(self, text: str, eos_token:str, answer: str = '') -> str:
        if answer != '':
            answer += "\n" + eos_token

        return f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{self.task_description}
{self.instructions}

### Input:
Text: {text}

### Response:
{answer}'''



class StudentEssayFormatter(BaseFormatter):
    def __init__(self):
        task_description = "Your task is to identify and list any personal information in the input text. Do not include information that is about famous people, only the author and people close to the author."
        instructions = '''Please, list the following personal information.

List of Names:
* first full or partial name of a person

List of Emails:
* email address of a person

List of Usernames:
* username of a person

List of ID Numbers:
* number or sequence of characters that could be used to identify a person like the student ID or a social security number

List of Phone Numbers:
* phone number associated with a person

List of URLs:
* URL that might be used to identify a person

List of Street Addresses:
* full or partial street address that is associated with the person, such as a home address'''
        super().__init__(task_description, instructions)

class MedicalDocumentFormatter(BaseFormatter):
    def __init__(self):
        task_description = "Your task is to identify and list any medical information in the input text."
        instructions = '''Please, list the following medical information.

List of Diagnoses:
* diagnosis names

List of Medications:
* medication names

List of Procedures:
* medical procedures'''
        super().__init__(task_description, instructions)

class ChainOfThought:
    def __init__(self, task_description: str, information_type: str, instructions = '', ):
        self.task_description = task_description
        if instructions != '':
            self.instructions = ' Additionally, '+instructions
        else:
            self.instructions = ''
        self.information_type = information_type

    def format_prompt(self, information_list):
        return f'''{self.task_description} Provided are several different types of {self.information_type} in a JSON format. Help me by giving an explanation for each one as to why they are considered {self.information_type} in detail. Also why it is the type of {self.information_type} it is classified as.{self.instructions} Fill in the reason section of each entry in the JSON file.
        [INFO_LIST]
        '''.replace('[INFO_LIST]', str(information_list))

class StandardAnswer:
    def __init__(self, task_description: str, information_type: str, instructions = '', ):
        self.task_description = task_description
        if instructions != '':
            self.instructions = ' Additionally, '+instructions
        else:
            self.instructions = ''
        self.information_type = information_type

    def format_prompt(self, information_list):
        return f'''{self.task_description} Provided are several different types of {self.information_type} in a JSON format.{self.instructions} Fill in the reason section of each entry in the JSON file.
        [INFO_LIST]
        '''.replace('[INFO_LIST]', str(information_list))

class StandardAnswer_StudentPII(StandardAnswer):
    def __init__(self):
        task_description = 'I am creating a dataset to help remove PII.'
        instructions = ''
        information_type = 'Personally Identifiable Information (PII)'

        super().__init__(task_description, information_type, instructions)

class StudentPII_COT(ChainOfThought):
    def __init__(self):
        task_description = 'I am creating a dataset to help remove PII.'
        instructions = ''
        information_type = 'Personally Identifiable Information (PII)'

        super().__init__(task_description, information_type, instructions)

