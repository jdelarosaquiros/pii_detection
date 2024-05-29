from SID_Dataset_Generator.dataset import Dataset, Assembler
from SID_Dataset_Generator.templates.templates import StudentEssayFormatter, StudentPII_COT
from SID_Dataset_Generator.generate_pii import main
import argparse

student_formatter = StudentEssayFormatter()
dataset_generator = Dataset(assembler=Assembler, formatter=student_formatter)

use_gen_data = False
if use_gen_data:
    data = main()
else:
    pii_test = [['"bryce hinkley" (name)','"123 south blvd, san antionio" (street_address)']]
    essays = ['Want a more difficult brutal game play experience? This is your mod then. I made various tweaks to everything to give it weight, slower levelling, more brutal debuff effects, no fast traveling, no healing from beds and many other things listed below. This mod is intended to be used with Hardcore mode on Very Hard difficulty in tandem with BLEED and SawyerBatty to give you an FWE like experience in Tale of Two Wastelands.']

print(dataset_generator.generate(essays, pii_test, StudentPII_COT, 'essay')[0]['answer'])

