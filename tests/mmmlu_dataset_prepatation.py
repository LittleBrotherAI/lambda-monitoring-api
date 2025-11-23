from datasets import load_dataset, concatenate_datasets

# Define your languages with their split names
languages = {
    'DE_DE': 'German',
    'FR_FR': 'French', 
    #'SW_KE': 'Swahili',
    'PT_BR': 'Brazilien Portuguese',
    'ZH_CN': 'Simplified Chinese'
}

subject = "elementary_mathematics"
n_samples = 5

all_samples = []

for lang_code, lang_name in languages.items():
    # Load the specific language split
    dataset = load_dataset("openai/MMMLU", lang_code, split="test")
    
    # Filter for elementary_mathematics
    filtered = dataset.filter(lambda x: x['Subject'] == subject)
    
    # Take top 20
    samples = filtered.select(range(min(n_samples, len(filtered))))
    
    # Add language identifier column
    samples = samples.add_column("language", [lang_code] * len(samples))
    
    all_samples.append(samples)
    print(f"{lang_name} ({lang_code}): {len(samples)} samples")

# Combine all datasets
combined_dataset = concatenate_datasets(all_samples)

print(f"\nTotal combined samples: {len(combined_dataset)}")
print(f"Columns: {combined_dataset.column_names}")

combined_dataset.to_csv("tests/mmmlu_elementary_math.csv")