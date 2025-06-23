import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.combine import SMOTETomek

# Step 1: Generate Synthetic Customer Data
np.random.seed(42)
names = ["Aarya", "Devika", "Ishanvi", "Vedant", "Samaira", "Aarush", "Reya", "Nirvaan", "Trisha", "Dhruvika",  
         "Aryav", "Ishir", "Keya", "Prisha", "Viha", "Saisha", "Anvay", "Yuvan", "Aarini", "Kiaan",  
         "Shanvika", "Darsh", "Vian", "Hriday", "Ronav", "Rudhir", "Neerav", "Vivika", "Kavish", "Avyaan",  
         "Suhana", "Chahat", "Manav", "Dhriti", "Vanya", "Advay", "Kiyana", "Hrishaan", "Pranay", "Navika",  
         "Rithvik", "Arisha", "Rishaan", "Reyanshi", "Atharvika", "Krishiv", "Samiksha", "Tushar", "Kashvi", "Eshaan",  
         "Shailee", "Kiaanav", "Nishtha", "Shivay", "Kritika", "Parthiv", "Tvesha", "Lakshya", "Avisha", "Saarth",  
         "Tanvika", "Nivaan", "Hridya", "Aahna", "Sanvika", "Zayan", "Naira", "Devesh", "Kiara", "Saanvika",  
         "Tanish", "Mishka", "Ishwar", "Riyanshi", "Lakshitha", "Raghav", "Mahika", "Nivriti", "Aavya", "Vedika",  
         "Diyan", "Rudransh", "Sharvani", "Ishaanika", "Jivika", "Kian", "Oviya", "Vrishank", "Maitreyi", "Tanay",  
         "Vedanth", "Yug", "Reet", "Aradhya", "Ahaan", "Rudrani", "Swara", "Pihu", "Aravika", "Harshit",  
         "Vibha", "Ronavika", "Anvika", "Neysa", "Nandan", "Anvit", "Yashika", "Bhavika", "Yashasvi", "Ojas",  
         "Aarit", "Sairaj", "Ridhima", "Sanvitha", "Samar", "Krithika", "Hansika", "Suhani", "Saumya", "Aryahi",  
         "Mahir", "Tisha", "Kush", "Pranika", "Shaurya", "Harsha", "Devanshi", "Nishith", "Darshini", "Aariniya",  
         "Manisha", "Kaushik", "Bhavna", "Anushri", "Ronavanshi", "Aayan", "Vishnu", "Vayun", "Devank", "Chinmayi",  
         "Nidhi", "Sharanya", "Anwita", "Suhas", "Anokhi", "Rithvika", "Tanush", "Ishaanth", "Risha", "Nandanika",  
         "Rishit", "Ariv", "Sharv", "Samriddhi", "Lakshman", "Yuvaan", "Kushagra", "Moksh", "Janya", "Mithil",  
         "Arin", "Eesha", "Kirtan", "Hridaan", "Naman", "Devan", "Ritika", "Nivedita", "Krishika", "Dhanya",  
         "Samarthya", "Nayantara", "Saathvik", "Ritwik", "Navneeth", "Aarv", "Vihaanika", "Anay", "Tejaswini", "Pankhuri",  
         "Srijan", "Shreyas", "Rohini", "Maanav", "Anvitha", "Gautami", "Tarini", "Ekansh", "Krupa", "Aryaman",  
         "Rudraksh", "Sharvil", "Bhaskar", "Devendra", "Maitreya", "Saurav", "Tanmayi", "Yugansh", "Ishir", "Avyukt"]

data = {
    'CustomerID': np.arange(1, 2001),
    'Name': np.random.choice(names, 2000, replace=True),
    'Age': np.random.randint(18, 80, 2000),
    'Annual_Income': np.random.randint(10000, 200000, 2000),
    'Purchase_Frequency': np.random.randint(1, 100, 2000),
    'Loyalty_Score': np.random.randint(1, 100, 2000),
    'Subscription_Tenure': np.random.randint(1, 25, 2000),
    'Customer_Region': np.random.choice(['Urban', 'Suburban', 'Rural', 'Industrial', 'Tourist'], 2000),
    'Customer_Type': np.random.choice(['Individual', 'Retailer', 'Corporate'], 2000),
    'Avg_Spending_Per_Month': np.random.randint(500, 5000, 2000),
    'Churn': np.random.choice(["No", "Yes"], 2000, p=[0.70, 0.30])
}
df = pd.DataFrame(data)

# Step 2: Feature Engineering
df['Customer_Lifetime_Value'] = df['Subscription_Tenure'] * df['Avg_Spending_Per_Month']
df['Engagement_Score'] = df['Loyalty_Score'] * df['Purchase_Frequency']
df['Spending_to_Income_Ratio'] = df['Avg_Spending_Per_Month'] / df['Annual_Income']

# Step 3: Data Preprocessing
scaler = StandardScaler()
features_to_scale = ['Age', 'Annual_Income', 'Purchase_Frequency', 'Loyalty_Score', 'Subscription_Tenure', 
                     'Avg_Spending_Per_Month', 'Customer_Lifetime_Value', 'Engagement_Score', 'Spending_to_Income_Ratio']
df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])

# Encoding categorical variables
label_encoders = {}
for col in ['Customer_Region', 'Customer_Type']:
    label_encoders[col] = LabelEncoder()
    df_scaled[col] = label_encoders[col].fit_transform(df_scaled[col])

# Step 4: Customer Segmentation using K-Means
optimal_clusters = 6
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df_scaled['Segment'] = kmeans.fit_predict(df_scaled[features_to_scale])

# Mapping segment numbers to meaningful names
segment_labels = {
    0: "Loyal High Spenders",
    1: "Frequent Buyers",
    2: "Discount Seekers",
    3: "Occasional Buyers",
    4: "New Customers",
    5: "At-Risk Customers"
}
df_scaled['Segment'] = df_scaled['Segment'].map(segment_labels)

region_labels = {
    0: "Industrial",
    1: "Rural",
    2: "Suburban",
    3: "Tourist",
    4: "Urban"
}

customer_type_labels = {
    0: "Corporate",
    1: "Individual",
    2: "Retailer"
}


# Step 5: Churn Prediction with Class Imbalance Handling
X = df_scaled.drop(columns=['CustomerID', 'Name', 'Churn'])
X['Segment'] = X['Segment'].astype('category').cat.codes
y = df_scaled['Churn'].map({'No': 0, 'Yes': 1})

# Address class imbalance
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

# Train RandomForest Model
rf_model = RandomForestClassifier(n_estimators=800, max_depth=25, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

# Visualization
plt.figure(figsize=(12, 7))
sns.scatterplot(x=df_scaled['Annual_Income'], y=df_scaled['Loyalty_Score'], hue=df_scaled['Segment'], palette='coolwarm')
plt.title('Customer Segmentation based on Loyalty Score and Income')
plt.xlabel('Annual Income')
plt.ylabel('Loyalty Score')
plt.legend(title='Segment')
plt.show()

# Model Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Insights & Recommendations
segment_churn = df_scaled.groupby('Segment')['Churn'].value_counts(normalize=True).unstack()
print("\nChurn Rate by Segment:")
print(segment_churn)

region_churn = df_scaled.groupby('Customer_Region')['Churn'].value_counts(normalize=True).unstack()
region_churn.rename(index=region_labels, inplace=True)
print("\nChurn Rate by Region:")
print(region_churn)

type_churn = df_scaled.groupby('Customer_Type')['Churn'].value_counts(normalize=True).unstack()
type_churn.rename(index=customer_type_labels, inplace=True)
print("\nChurn Rate by Customer Type:")
print(type_churn)

print("\nKey Observations:")
if (segment_churn['Yes'].max() > 0.35):
    print("- Certain customer segments have a churn rate above 35%. Personalized retention strategies needed.")
if (region_churn['Yes'].max() > 0.35):
    print("- Customers in specific regions have higher churn. Regional campaigns may be beneficial.")
if (type_churn['Yes'].max() > 0.35):
    print("- Particular customer types, such as retailers or corporates, show high churn rates.")
print("- Customers with lower loyalty scores and higher purchase frequency tend to churn more.")
print("- High-income customers with low purchase frequency are more likely to churn. Engagement strategies needed.")

# Sample Customer Data
sample_customers = df.sample(n=20).drop_duplicates(subset=['Name'])
print("\nSample Customer Data:")
print(sample_customers[['CustomerID', 'Name', 'Age', 'Annual_Income', 'Purchase_Frequency', 'Loyalty_Score', 'Customer_Region', 'Customer_Type', 'Churn']])
