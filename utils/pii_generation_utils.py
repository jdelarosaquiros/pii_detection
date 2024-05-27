import random
import re
from wonderwords import RandomWord

rand_word_generator = RandomWord()

pii_labels = ['STUDENT_NAME', 'EMAIL', 'ID_NUMBER', 'USERNAME', 'PERSONAL_URL', 'STREET_ADDRESS', 'PHONE_NUMBER']
valid_email_symbols = ['.', '_', '-', ''] # According to the internet, there are more valid symbols but these are the most common ones
valid_id_number_symbols = ['-', '_', ' ', '']
valid_phone_number_symbols = ['-', '_', '.', ' ', '']
valid_street_address_symbols = [' ', ', ']
street_abrivations = ['St.', 'Ave.', 'Blvd.', 'Rd.', 'Dr.', 'Ct.', 'Ln.', 'Pkwy.', 'Pl.', 'Ter.', 'Trl.', 'Way.', 'Hwy.']
appartment_abrivations = ['Apt.', 'Bldg.', 'Dept.', 'Fl.', 'Ste.', 'Unit.']
protocols = ['http://', 'https://', 'ftp://', 'ftps://']
subdomains = ['www.']
valid_url_symbols = ['$', '.', '!', '*']
valid_url_separators = ['_', '-', '.','+', '']

# Convert states initials to full names
state_abbr_to_full = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'DC': 'Washington D.C.',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming'
}

def generate_name_samples(first, mi, last, name) -> list[str]:
    rand_num = random.randint(1, 5)
    where_add_middle = random.randint(0, 2)

    # Randomly choose whether to change case of each variable
    if random.choice([True, False]):
        first = first.lower() if random.choice([True, False]) else first.upper() # Randomly choose whether to lowercase or uppercase
    
    if random.choice([True, False]):
        mi = mi.lower() if random.choice([True, False]) else mi.upper() # Randomly choose whether to lowercase or uppercase

    if random.choice([True, False]):
        last = last.lower() if random.choice([True, False]) else last.upper() # Randomly choose whether to lowercase or uppercase
    
    if random.choice([True, False]):
        name = name.lower() if random.choice([True, False]) else name.upper() # Randomly choose whether to lowercase or uppercase

    data = name.split(' ')
    first_name = data[0]
    last_name = data[1]

    # Randomly whether to add period to middle initial
    if random.choice([True, False]):
        mi = f"{mi}."

    # Randomly choose whether to add middle initial to first name, first, or neither
    if where_add_middle == 1:
        first = f"{first} {mi}"
    elif where_add_middle == 2:
        first_name = f"{first_name} {mi}"
    

    # Randomly choose whether to swap first and last name
    if random.choice([True, False]):
        temp = first_name
        first_name = last_name
        last_name = temp

        temp = first
        first = last
        last = temp
    
    name = f"{first_name} {last_name}"

    if rand_num == 1:
        return [first, last, name]
    elif rand_num == 2:
        return [first, f"{name} {last}"]
    elif rand_num == 3:
        return [last, f"{first} {name}"]
    elif rand_num == 4:
        return [name, f"{first} {last}"]
    else:
        return [f"{first} {name} {last}"]

def generate_email_samples(first, last, email) -> list[str]:
    rand_num = random.randint(1, 4)
    sep_chars = random.choices(valid_email_symbols, k=6)
    rand_word = rand_word_generator.word().capitalize()

    data = email.split('@')
    email_name = data[0]
    email_domain = data[1]

    # Randomly choose whether to change case of each variable
    if random.choice([True, False]):
        first = first.lower() if random.choice([True, False]) else first.upper() # Randomly choose whether to lowercase or uppercase

    if random.choice([True, False]):
        last = last.lower() if random.choice([True, False]) else last.upper() 
    
    if random.choice([True, False]):
        rand_word = rand_word.lower() if random.choice([True, False]) else rand_word.upper()
   
    # Randomly choose whether to add digits
    if random.choice([True, False]):
        where_add_digits = random.randint(1, 4) # Decide where to add digits
        digits = ''.join(random.choices('0123456789', k=random.randint(1, 5))) # Generate random digits
    else:
        where_add_digits = 0

    if where_add_digits == 1: 
        if random.choice([True, False]): # Randomly choose whether to append or prepend
            first = f"{first}{digits}"
        else:
            first = f"{digits}{first}"
    elif where_add_digits == 2:
        if random.choice([True, False]):
            last = f"{last}{digits}"
        else:
            last = f"{digits}{last}"
    elif where_add_digits == 3:
        if random.choice([True, False]):
            rand_word = f"{rand_word}{digits}"
        else:
            rand_word = f"{digits}{rand_word}"
    elif where_add_digits == 4:
        if random.choice([True, False]):
            email_name = f"{email_name}{digits}"
        else:
            email_name = f"{digits}{email_name}"

    # Randomly choose whether to swap digits
    if random.choice([True, False]):
        where_to_swap = random.choices(range(1, 7), k=random.randint(1, 7)) # Decide where to swap digits
    else:
        where_to_swap = []

    # Randomly sawp order of first, last, word, and email_name
    if 1 in where_to_swap:
        temp = first
        first = last
        last = temp

    if 2 in where_to_swap:
        temp = first
        first = rand_word
        last = temp

    if 3 in where_to_swap:
        temp = first
        first = rand_word
        rand_word = temp
    
    if 4 in where_to_swap:
        temp = last
        last = rand_word
        rand_word = temp

    if 5 in where_to_swap:
        temp = first
        first = email_name
        email_name = temp

    if 6 in where_to_swap:
        temp = last
        last = email_name
        email_name = temp

    if 7 in where_to_swap:
        temp = rand_word
        rand_word = email_name
        email_name = temp
    
    # Randomly choose whether use the same separator character
    if random.choice([True, False]):
        sep_chars = [sep_chars[0] for _ in sep_chars]
    
    if rand_num == 1:
        return [f"{email_name}@{email_domain}", f"{first}@{email_domain}", f"{last}@{email_domain}", f"{rand_word}@{email_domain}"]
    elif rand_num == 2:
        return [f"{first}{sep_chars[0]}{email_name}@{email_domain}", f"{last}{sep_chars[1]}{email_name}@{email_domain}", f"{rand_word}{sep_chars[2]}{email_name}@{email_domain}"]
    elif rand_num == 3:
        return [f"{first}{sep_chars[0]}{last}{sep_chars[1]}{email_name}@{email_domain}", f"{rand_word}{sep_chars[2]}{first}{sep_chars[3]}{email_name}@{email_domain}", f"{last}{sep_chars[4]}{rand_word}{sep_chars[5]}{email_name}@{email_domain}"]
    elif rand_num == 4:
        return [f"{first}{sep_chars[0]}{last}{sep_chars[1]}{rand_word}{sep_chars[2]}{email_name}@{email_domain}"]


def generate_username_samples(name, email) -> list[str]:
    rand_num = random.randint(1, 4)
    sep_chars = random.choices(valid_email_symbols, k=6)
    rand_word = rand_word_generator.word().capitalize()

    email_name = email.split('@')[0]

    data = name.split(' ')
    first = data[0]
    last = data[1]

     # Randomly choose whether to change case of each variable
    if random.choice([True, False]):
        first = first.lower() if random.choice([True, False]) else first.upper() # Randomly choose whether to lowercase or uppercase

    if random.choice([True, False]):
        last = last.lower() if random.choice([True, False]) else last.upper() 
    
    if random.choice([True, False]):
        rand_word = rand_word.lower() if random.choice([True, False]) else rand_word.upper()
   
    
    # Randomly choose whether to add digits
    if random.choice([True, False]):
        where_add_digits = random.randint(1, 4) # Decide where to add digits
        digits = ''.join(random.choices('0123456789', k=random.randint(1, 5))) # Generate random digits
    else:
        where_add_digits = 0

    if where_add_digits == 1: 
        if random.choice([True, False]): # Randomly choose whether to append or prepend
            first = f"{first}{digits}"
        else:
            first = f"{digits}{first}"
    elif where_add_digits == 2:
        if random.choice([True, False]):
            last = f"{last}{digits}"
        else:
            last = f"{digits}{last}"
    elif where_add_digits == 3:
        if random.choice([True, False]):
            rand_word = f"{rand_word}{digits}"
        else:
            rand_word = f"{digits}{rand_word}"
    elif where_add_digits == 4:
        if random.choice([True, False]):
            email_name = f"{email_name}{digits}"
        else:
            email_name = f"{digits}{email_name}"

    # Randomly choose whether to swap variables
    if random.choice([True, False]):
        where_to_swap = random.choices(range(1, 7), k=random.randint(1, 7)) # Decide where to swap digits
    else:
        where_to_swap = []

    # Randomly sawp order of first, last, word, and email_name
    if 1 in where_to_swap:
        temp = first
        first = last
        last = temp

    if 2 in where_to_swap:
        temp = first
        first = rand_word
        last = temp

    if 3 in where_to_swap:
        temp = first
        first = rand_word
        rand_word = temp
    
    if 4 in where_to_swap:
        temp = last
        last = rand_word
        rand_word = temp

    if 5 in where_to_swap:
        temp = first
        first = email_name
        email_name = temp

    if 6 in where_to_swap:
        temp = last
        last = email_name
        email_name = temp

    if 7 in where_to_swap:
        temp = rand_word
        rand_word = email_name
        email_name = temp
    
    # Randomly choose whether use the same separator character
    if random.choice([True, False]):
        sep_chars = [sep_chars[0] for _ in sep_chars]

    # Check whether first, last, and words do not contain digits and add digits if they do not to prevent normal names and words from being used as usernames
    if rand_num == 1:
        if not any(char.isdigit() for char in first):
            digits = ''.join(random.choices('0123456789', k=random.randint(1, 6)))
            first = f"{first}{digits}"
        
        if not any(char.isdigit() for char in last):
            digits = ''.join(random.choices('0123456789', k=random.randint(1, 6)))
            last = f"{last}{digits}"

        if not any(char.isdigit() for char in rand_word):
            digits = ''.join(random.choices('0123456789', k=random.randint(1, 6)))
            rand_word = f"{rand_word}{digits}"

    if rand_num == 1:
        return [f"{email_name}", f"{first}", f"{last}", f"{rand_word}"]
    elif rand_num == 2:
        return [f"{first}{sep_chars[0]}{email_name}", f"{last}{sep_chars[1]}{email_name}", f"{rand_word}{sep_chars[2]}{email_name}"]
    elif rand_num == 3:
        return [f"{first}{sep_chars[0]}{last}{sep_chars[1]}{email_name}", f"{rand_word}{sep_chars[2]}{first}{sep_chars[3]}{email_name}", f"{last}{sep_chars[4]}{rand_word}{sep_chars[5]}{email_name}"]
    elif rand_num == 4:
        return [f"{first}{sep_chars[0]}{last}{sep_chars[1]}{rand_word}{sep_chars[2]}{email_name}"]
    
def generate_id_number_samples(digit) -> str:
    is_split = random.choice([True, False])
    max_splits = len(digit) - 1
    num_separaors = random.randint(1, max_splits)
    sep_char = random.choice(valid_id_number_symbols)

    if not is_split:
        return digit
    
    split_indices = random.sample(range(1, len(digit)), num_separaors)

    for i in split_indices:
        digit = digit[:i] + sep_char + digit[i:]
        
    return digit

def generate_phone_number_samples(phone) -> str:
    rand_num = random.randint(1, 5)
    sep_chars = random.choices(valid_phone_number_symbols, k=3)

    # Split phone number into area code, first three digits, and last four digits
    processed_phone = re.sub(r'[)(]', '', phone)
    processed_phone = re.sub(r'-', ' ', processed_phone)
    data = processed_phone.split(' ')
    area_code = data[0]
    first_three = data[1]
    last_four = data[2]

    # Randomly choose whether to choose whehter to add digits
    if random.choice([True, False]):
        digits = ''.join(random.choices('0123456789', k=random.randint(1, 3)))
        where_add_digits = random.randint(1, 3)
    else:
        where_add_digits = 0
    
    if where_add_digits == 1:
        if random.choice([True, False]):
            area_code = f"{area_code}{digits}"
        else:
            area_code = f"{digits}{area_code}"
    elif where_add_digits == 2:
        if random.choice([True, False]):
            first_three = f"{first_three}{digits}"
        else:
            first_three = f"{digits}{first_three}"
    elif where_add_digits == 3:
        if random.choice([True, False]):
            last_four = f"{last_four}{digits}"
        else:
            last_four = f"{digits}{last_four}"

    # Randomly choose whether use the same separator character
    if random.choice([True, False]):
        sep_chars = [sep_chars[0] for _ in sep_chars]
    
    # Randomly choose whether to include international code
    if random.choice([True, False]):
        digits = ''.join(random.choices('0123456789', k=random.randint(1, 4)))
        inter_code = f"{random.choice(['+', ''])}{digits}{sep_chars[0]}"
    else:
        inter_code = ""

    # Randomly choose whether to wrap area code in parentheses
    if random.choice([True, False]):
        area_code = f" ({area_code}) "
        inter_code.replace(sep_chars[0], '') # Remove separator character if area code is wrapped in parentheses
    else:
        area_code = f"{area_code}{sep_chars[1]}"
        

    first_three = f"{first_three}{sep_chars[2]}"

    return f"{inter_code}{area_code}{first_three}{last_four}"

def generate_street_address_samples(street, state, zip, zip9) -> list[str]:
    rand_num = random.randint(1, 5)
    sep_chars = random.choices(valid_id_number_symbols, k=4)
    apt_num = ''.join(random.choices('0123456789', k=random.randint(1, 5))) # Generate random apartment number
    
    #Randomly choose whether to use state initials or full name
    if random.choice([True, False]):
        state = state_abbr_to_full[state]

    # Randomly choose whether to add abbreviations and prepend or append it
    if random.choice([True, False]):
        street = f"{street} {random.choice(street_abrivations)}" if random.choice([True, False]) else f"{random.choice(street_abrivations)} {street}" 

    # Add abrivation to apartment number by prepending or appending it
    apt_num = f"{apt_num} {random.choice(appartment_abrivations)}" if random.choice([True, False]) else f"{random.choice(appartment_abrivations)} {apt_num}"

    # Randomly choose whether to alter each variable
    if random.choice([True, False]):
        street = street.lower() if random.choice([True, False]) else street.upper() # Randomly choose whether to lowercase or uppercase
    
    if random.choice([True, False]):
        state = state.lower() if random.choice([True, False]) else state.upper()

    if random.choice([True, False]):
        apt_num = apt_num.lower() if random.choice([True, False]) else apt_num.upper()
    
    # Randomly choose whether to add street number to street
    if random.choice([True, False]):
        street_number = ''.join(random.choices('0123456789', k=random.randint(1, 6)))
        street = f"{street_number} {street}"

    # Randomly choose whether use the same separator character
    if random.choice([True, False]):
        sep_chars = [sep_chars[0] for _ in sep_chars]
    
    # Randomly choose whether to use zip9
    if random.choice([True, False]):
        zip = zip9

    # Randomly choose whether to add zip code to state
    if random.choice([True, False]):
        state = f"{state}{sep_chars[-1]}{zip}"

    # Randomly choose whether to swap order of street, apt_num, state, and zip and ranmdomly swap them
    if random.choice([True, False]):
        where_to_swap = random.choices(range(1, 6), k=random.randint(1, 6)) # Decide where to swap digits
    else:
        where_to_swap = []
    
    if 1 in where_to_swap:
        temp = street
        street = apt_num
        apt_num = temp
    
    if 2 in where_to_swap:
        temp = street
        street = state
        state = temp
    
    if 4 in where_to_swap:
        temp = apt_num
        apt_num = state
        state = temp

    # Randomly choose whether to include each variable
    if random.choice([True, False]):
        processed_street = f"{street}{sep_chars[0]}"
    else:
        processed_street = ""

    if random.choice([True, False]):
        processed_apt_num = f"{apt_num}{sep_chars[1]}"
    else:
        processed_apt_num = ""
    
    if random.choice([True, False]):
        processed_state = f"{state}"
    else:
        processed_state = ""

    
    # Randomly choose whether to return each variable separately or together
    if random.choice([True, False]):
        return [f"{street}", f"{apt_num}", f"{state}"]
    else:
        return [f"{processed_street}{processed_apt_num}{processed_state}"]

def format_url_path(rand_strings) -> str:
    sep_chars = random.choices(valid_url_separators, k = len(rand_strings) - 1)

    # For each string randomly choose whether to append or prepend a symbol or do neither
    if random.choice([True, False]):
        for i, rand_string in enumerate(rand_strings):
            if random.choice([True, False]):
                rand_strings[i] = random.choice(valid_url_symbols) + rand_string
            elif random.choice([True, False]):
                rand_strings[i] = rand_string + random.choice(valid_url_symbols)

    # Randomly choose whether use the same separator character
    if random.choice([True, False]):
        sep_chars = [sep_chars[0] for _ in sep_chars]

    # Randomly choose whether to dd separator character to each string
    rand_strings = [rand_strings[i] + sep_char for i, sep_char in enumerate(sep_chars)]

    return ''.join(rand_strings)
        

def format_arguments(strings) -> str:
    for i, rand_string in enumerate(strings):
        if i + 1 >= len(strings):
            break

        if i % 2 == 0:
            strings[i] = f"{rand_string}={strings[i+1]}"
            strings[i+1] = ''
        
    return '&'.join(strings)

def generate_rand_strings(num_strings) -> list[str]:
    rand_strings = []

    for _ in range(num_strings):
        # Randomly choose whether to use a random word or a random number
        if random.choice([True, False]):
            rand_strings.append(rand_word_generator.word())
        else:
            rand_strings.append(''.join(random.choices('0123456789', k=random.randint(1, 10))))

     # Randomly choose whether to change case of each variable
    if random.choice([True, False]):
        # Randomly choose whether to keep the casing the same or different
        is_same = random.choice([True, False])
        if is_same:
            rand_num = random.randint(1, 3)

        for i, rand_string in enumerate(rand_strings):
            if not is_same:
                rand_num = random.randint(1, 3)

            # Only case first letter
            if rand_num == 1:
                rand_strings[i] = rand_string.capitalize()
            
            # Lowercase all letters
            elif rand_num == 2:
                rand_strings[i] = rand_string.lower()

            # Uppercase all letters
            elif rand_num == 3:
                rand_strings[i] = rand_string.upper()

    return rand_strings

def generate_url_samples(domain) -> str:
    rand_num = random.randint(0, 2)
    rand_depth = random.randint(0, 5)
    protocol = ''
    subdomain = ''
    path_list = []

    # Randomly choose whether to add a protocol
    if random.choice([True, False]):
        protocol = random.choice(protocols)

    # Randomly choose whether to add a subdomain
    if random.choice([True, False]):
        subdomain = random.choice(subdomains) if random.choice([True, False]) else rand_word_generator.word() + '.'

    # Randomly choose whether to add a path
    if rand_depth > 0:
        for rand_depth in range(rand_depth):
            rand_num_string = random.randint(1, 5)
            rand_strings = generate_rand_strings(rand_num_string)

            path_list.append(format_url_path(rand_strings))
    
    # Randomly decide whether to add arguments, a path separator, or nothing
    if rand_num == 1:
        rand_num_string = random.randint(1, 5)
        rand_strings = generate_rand_strings(rand_num_string)

        path_list.append('?' + format_arguments(rand_strings))

    elif rand_num == 2:
        path_list.append('/')

    if path_list:
        path = '/' + '/'.join(path_list)
    else:
        path = ''

    return f"{protocol}{subdomain}{domain}{path}"