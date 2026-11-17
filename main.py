
# =====================================================
# MARKET BASKET ANALYSIS USING APRIORI ALGORITHM
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load Dataset
data = pd.read_csv("dataset.csv")

print("\nMARKET BASKET ANALYSIS")
print("======================")

print("\nTotal Records:", len(data))

# Step 2: Data Preprocessing
data["Date"] = pd.to_datetime(
    data["Date"], format="%d-%m-%Y"
)

data["Transaction"] = (
    data["Member_number"].astype(str)
    + "_"
    + data["Date"].astype(str)
)

transactions = data.groupby("Transaction")[
    "itemDescription"
].apply(list).tolist()

print("Total Transactions:", len(transactions))
print("Total Unique Items:", data["itemDescription"].nunique())

# Step 3: Transaction Encoding
encoder = TransactionEncoder()

encoded_data = encoder.fit(transactions).transform(transactions)

basket = pd.DataFrame(
    encoded_data,
    columns=encoder.columns_
)

# Step 4: Apply Apriori Algorithm
frequent_items = apriori(
    basket,
    min_support=0.005,
    use_colnames=True
)

print("\nTOP FREQUENT ITEMSETS")
print("---------------------")

print(
    frequent_items.sort_values(
        by="support", ascending=False
    ).head(10).to_string(index=False)
)

# Step 5: Generate Association Rules
if len(frequent_items) > 1:
    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=0.1
    )
else:
    rules = pd.DataFrame()

# Step 6: Filter Strong Association Rules
if not rules.empty:
    strong_rules = rules[rules["lift"] > 1].copy()

    strong_rules = strong_rules.sort_values(
        by="lift", ascending=False
    )
else:
    strong_rules = pd.DataFrame()

print("\nSTRONG ASSOCIATION RULES")
print("------------------------")

if not strong_rules.empty:
    print(
        strong_rules[
            ["antecedents", "consequents",
             "support", "confidence", "lift"]
        ].head(10).to_string(index=False)
    )
else:
    print("No strong positive association rules found.")

# Step 7: Save Analysis Results
frequent_items.to_csv("frequent_itemsets.csv", index=False)

rules.to_csv("association_rules.csv", index=False)

strong_rules.to_csv(
    "strong_association_rules.csv", index=False
)

# Step 8: Prepare Top 10 Frequent Items
single_items = frequent_items[
    frequent_items["itemsets"].apply(len) == 1
].copy()

single_items["item"] = single_items["itemsets"].apply(
    lambda x: list(x)[0]
)

top_items = single_items.nlargest(10, "support")

# Step 9: Visualize Frequent Items
plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_items,
    x="support",
    y="item"
)

plt.title("Top 10 Frequently Purchased Items")
plt.xlabel("Support")
plt.ylabel("Grocery Items")
plt.tight_layout()

plt.savefig("frequent_items.png")
plt.show()

# Step 10: Visualize Association Rules
if not rules.empty:
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=rules,
        x="support",
        y="confidence",
        hue="lift",
        size="lift",
        sizes=(50, 300)
    )

    plt.title("Association Rules Analysis")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.tight_layout()

    plt.savefig("association_rules.png")
    plt.show()

# Step 11: Visualize Strong Rules
if not strong_rules.empty:
    top_rules = strong_rules.head(10).copy()

    top_rules["Rule"] = top_rules.apply(
        lambda row: (
            ", ".join(row["antecedents"])
            + " -> "
            + ", ".join(row["consequents"])
        ),
        axis=1
    )

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=top_rules,
        x="lift",
        y="Rule"
    )

    plt.title("Top Strong Association Rules")
    plt.xlabel("Lift")
    plt.ylabel("Product Associations")
    plt.tight_layout()

    plt.savefig("strong_rules.png")
    plt.show()

# Step 12: Final Summary
print("\nPROJECT SUMMARY")
print("----------------")

print("Frequent Itemsets Found:", len(frequent_items))
print("Association Rules Found:", len(rules))
print("Strong Rules Found:", len(strong_rules))

print("\nAll Analysis Results Saved Successfully!")
print("MARKET BASKET ANALYSIS COMPLETED!")
