# Market Basket Analysis Using Apriori Algorithm

## Project Overview

Market Basket Analysis is a data mining technique used to identify products that are frequently purchased together by customers.

This project uses the Apriori Algorithm to analyze grocery shopping transactions and discover frequent itemsets and association rules.

## Objectives

- Analyze grocery shopping transactions.
- Identify frequently purchased products.
- Discover relationships between different products.
- Generate association rules using the Apriori Algorithm.
- Understand customer purchasing patterns.

## Technologies Used

- Python
- Pandas
- Mlxtend
- Matplotlib
- Seaborn
- Visual Studio Code

## Dataset Description

This project uses a grocery transactions dataset containing the following columns:

| Column | Description |
|---|---|
| Member_number | Customer identification number |
| Date | Purchase date |
| itemDescription | Product purchased |

The dataset contains 38,765 records and 167 unique products.

Transactions are created by grouping products purchased by the same customer on the same date.

## Algorithm Used

### Apriori Algorithm

Apriori is an unsupervised learning algorithm used to find frequent itemsets in transactional datasets.

It identifies frequently occurring product combinations and generates association rules.

### Evaluation Measures

- **Support:** Measures how frequently an item or item combination appears in all transactions.
- **Confidence:** Measures how often the consequent item is purchased when the antecedent item is purchased.
- **Lift:** Measures the strength of association between two items compared with independent purchases.

## Project Workflow

1. Load the grocery dataset.
2. Inspect and preprocess the data.
3. Group products into customer transactions.
4. Convert transactions into binary format.
5. Apply the Apriori algorithm.
6. Generate association rules.
7. Identify positive associations using lift.
8. Visualize the results.
9. Export the results as CSV files and PNG graphs.

## Project Structure

```text
Market-Basket-Analysis/
│
├── dataset.csv
├── main.py
├── README.md
│
└── outputs/
    ├── frequent_itemsets.csv
    ├── association_rules.csv
    ├── strong_association_rules.csv
    ├── frequent_items.png
    ├── association_rules.png
    └── strong_rules.png
```

## How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install pandas mlxtend matplotlib seaborn
```

### Step 2: Run the Python Program

```bash
python main.py
```

## Output Files

The project generates:

- **frequent_itemsets.csv** – Contains frequently purchased item combinations.
- **association_rules.csv** – Contains generated association rules.
- **strong_association_rules.csv** – Contains rules with lift greater than 1.
- **frequent_items.png** – Visualization of the most frequently purchased items.
- **association_rules.png** – Visualization of association rules.
- **strong_rules.png** – Visualization of strong association rules, when available.

## Conclusion

This project demonstrates the application of the Apriori Algorithm in Market Basket Analysis. It helps identify frequently purchased products and discover associations between items in grocery transactions.

The findings can help retailers understand customer purchasing behavior and make better decisions regarding product placement and marketing strategies.

## Future Enhancements

- Develop a web-based interface.
- Use larger transaction datasets.
- Improve association rule filtering.
- Build a product recommendation system.
