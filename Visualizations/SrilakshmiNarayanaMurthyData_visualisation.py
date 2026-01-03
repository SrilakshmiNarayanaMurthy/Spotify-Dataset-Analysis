#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_rows", 20)


# # Load Data

# In[3]:


penguin = pd.read_csv(r"C:\Users\Srilakshmi N Murthy\Desktop\MS 2nd sem\Data visualization\penguins.csv")


# In[4]:


penguin.head(20)


# # Overview

# In[5]:


print("Rows, columns", penguin.shape)


# In[6]:


penguin.info()


# In[7]:


penguin.isna().mean().sort_values(ascending=False).head(10).to_frame("missing_column")


# # FIX DTYPES

# In[8]:


penguin.dtypes.head(18)


# In[9]:


#converting strings to categories

cat_cols = ("studyName", "Species", "Region", "Island", "Stage", "Individual ID", "Clutch Completion", "Sex", "Comments")
for x in cat_cols:
    if x in penguin:
        penguin[x] = penguin[x].astype("category")


# In[10]:


penguin.dtypes.head(17)


# # handling duplicates

# In[11]:


penguin.shape


# In[12]:


penguin= penguin.drop_duplicates()
print("After dropping duplicates", penguin.shape)


# In[13]:


penguin.duplicated().sum() #no duplicates


# # Missing values

# In[14]:


missing_column = penguin.isna().mean()
mostly_missing = missing_column[missing_column > 0.6].index.tolist()
penguin_clean = penguin.drop(columns = mostly_missing)
print("Dropped mostly missing values",mostly_missing)


# In[15]:


num_cols = penguin_clean.select_dtypes(include = "number").columns
cat_cols = penguin_clean.select_dtypes(include = "category").columns

penguin_clean[num_cols]=penguin_clean[num_cols].fillna(penguin_clean[num_cols].median())

for c in cat_cols:
    penguin_clean=penguin_clean.ffill().bfill()
    
penguin_clean.isna().sum()


# In[16]:


penguin_clean.isna().sum().sum()


# ## Every listed column has 0 missing values.
# #So your penguin dataset is already perfectly clean in terms of NaNs.

# Standardize text by lowercasing, removing unwanted characters, trimming whitespace, and applying consistent casing. This avoids problems like treating "Southampton", "southampton", and " Southampton" as different categories.

# # Text cleaning

# In[17]:


for col in cat_cols:
    if col in penguin_clean.columns:
        penguin_clean[col + "_norm"]= (
            penguin_clean[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+"," ",regex=True)
            .str.title()
        )
        


# In[ ]:





# In[18]:


penguin_clean = penguin_clean.rename(columns={
    "Culmen Length (mm)": "Culmen_Length_mm",
    "Culmen Depth (mm)": "Culmen_Depth_mm",
    "Flipper Length (mm)": "Flipper_Length_mm",
    "Body Mass (g)": "Body_Mass_g",
    "Delta 15 N (o/oo)": "Delta15N_o_per_oo",
    "Delta 13 C (o/oo)": "Delta13C_o_per_oo"
})


# In[19]:


cols_to_show = [c for c in penguin_clean.columns]
penguin_clean[cols_to_show].head(8)


# # Outliers

# In[20]:


p99 = penguin_clean["Body_Mass_g"].quantile(0.99)
penguin_clean["Body_Mass_g_capped"]=penguin_clean["Body_Mass_g"].clip(upper=p99)
penguin_clean[["Body_Mass_g","Body_Mass_g_capped"]].describe()


# # Feature Engineering

# In[21]:


penguin_clean["Body Mass(kg)"] = penguin_clean["Body_Mass_g"]/1000

penguin_clean[["Body Mass(kg)"]]


# In[22]:


penguin_clean["Flipper_per_Culmen"] = penguin_clean["Flipper_Length_mm"]/penguin_clean["Culmen_Depth_mm"]
penguin_clean[["Flipper_per_Culmen"]]


# # Groupby and aggregate
# Comput average survival rates grouped by class and sex. GroupBy operations are central to data prep, letting us summarize and check trends quickly.
# penguin_clean.groupby("Species") → creates a grouped object (by species).
# 
# ["Body_Mass_g"] → tells pandas which numeric column you want to aggregate (take the mean of).

# In[23]:


#Average body mass per species
Average_body_mass_per_species=(
    penguin_clean
    .groupby("Species")["Body_Mass_g"]
    .mean()
    .rename("body_mass")
    .reset_index()
    .sort_values(["Species"])
)
Average_body_mass_per_species


# In[24]:


avg_mass_species_sex = (
    penguin_clean
      .groupby(["Species", "Sex"])["Body_Mass_g"]
      .mean()
      .reset_index(name="mean_mass_g")
)
avg_mass_species_sex


# #for my reference
# #A pivot table is a way to summarize data in a grid by grouping rows and columns.
# #Key difference
# 
# #Pivot = make data wide (good for reports/tables).
# 
# #Melt = make data long (good for charts/visualization).
# pd.pivot_table(
#     data,
#     index="row_group",
#     columns="column_group",
#     values="what_to_summarize",
#     aggfunc="how_to_summarize"
# )
# 

# # Pivot tables

# In[ ]:





# In[25]:


pivot_table = avg_mass_species_sex.pivot(
    index="Species",
    columns="Sex",
    values="mean_mass_g"
).round(3)

pivot_table


# # joins

# In[26]:


avg_mass = penguin_clean.groupby("Species")["Body_Mass_g"].mean().reset_index(name ="avg_mass")
avg_flipper=penguin_clean.groupby("Species")["Flipper_Length_mm"].mean().reset_index(name="avg_flipper")
merged = pd.merge(avg_mass, avg_flipper, on = "Species", how = "left")
merged


# # Time series reshape

# In[27]:


wide = penguin_clean.pivot_table(index = "Date Egg", columns = "Species", values= "Body_Mass_g")
long_again = wide.reset_index().melt(id_vars="Date Egg", var_name = "Species", value_name= "Body_Mass_g")
long_again.head()


# # Validation checks

# In[28]:


assert penguin_clean["Body_Mass_g"].ge(0).all()
assert penguin_clean["Flipper_Length_mm"].ge(0).all()
print("Basic checks passed ✅")


# # Plotting

# # Line Chart

# In[29]:


import matplotlib.pyplot as plt
avg_mass = penguin_clean.groupby("Species")["Body_Mass_g"].mean().reset_index()

fig, ax = plt.subplots()
ax.plot(avg_mass["Species"], avg_mass["Body_Mass_g"], marker="o", linewidth=4)

ax.set_title("Average Penguin Body Mass by Species")
ax.set_xlabel("Species")
ax.set_ylabel("Body_Mass_g")

plt.tight_layout()  #automatically adjusts the spacing between subplots and around the edges so everything fits nicely.
plt.xticks(rotation=45) #rotates x labels to 45 as the names are too big
plt.show()

The penguin dataset shows a clear distinction in body mass across species. Adelie and Chinstrap penguins have comparable average body masses (~3.7 kg), while Gentoo penguins are significantly heavier (~5.1 kg). This makes body mass a reliable feature for differentiating Gentoo penguins from the other two species.
# # Bar Graph

# In[30]:


import matplotlib.pyplot as plt

# Group and take the mean of each measurement per species
avg_traits = penguin_clean.groupby("Species")[["Body_Mass_g",
                                               "Flipper_Length_mm",
                                               "Culmen_Length_mm"]].mean()

# Now plot the DataFrame
ax = avg_traits.plot(kind="bar", figsize=(10,6))

ax.set_title("Average Penguin Measurements by Species")
ax.set_ylabel("Average Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[31]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.hist(penguin_clean["Body_Mass_g"], bins=20, color="#66c2a5", edgecolor="black")

plt.title("Distribution of Penguin Body Mass")
plt.xlabel("Body Mass (g)")
plt.ylabel("Number of Penguins")

plt.tight_layout()
plt.show()


# In[32]:


import seaborn as sns

plt.figure(figsize=(8,5))
sns.boxplot(data=penguin_clean, x="Species", y="Flipper_Length_mm", hue="Sex")

plt.title("Flipper Length by Species and Sex")
plt.xlabel("Species")
plt.ylabel("Flipper Length (mm)")
plt.tight_layout()
plt.show()


# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.regplot(
    data=penguin_clean,
    x="Flipper_Length_mm",
    y="Body_Mass_g",
    scatter_kws={"alpha":0.6},   # make points semi-transparent
    line_kws={"color":"red"}     # regression line color
)

plt.title("Regression: Flipper Length vs Body Mass")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.tight_layout()
plt.show()


# In[34]:


import pandas as pd
import matplotlib.pyplot as plt

island_species = pd.crosstab(penguin_clean["Island"], penguin_clean["Species"])

island_species.plot(kind="bar", stacked=True, figsize=(8,5))
plt.title("Penguin Species Distribution by Island")
plt.xlabel("Island")
plt.ylabel("Number of Penguins")
plt.legend(title="Species")
plt.tight_layout()
plt.show()


# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
num_cols = ["Body_Mass_g", "Flipper_Length_mm", "Culmen_Length_mm", "Culmen_Depth_mm"]
corr = penguin_clean[num_cols].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Heatmap of Penguin Measurements")
plt.tight_layout()
plt.show()


# In[37]:


import matplotlib.pyplot as plt

# Create figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# First scatter plot: Flipper Length vs Body Mass
ax1.scatter(
    penguin_clean["Flipper_Length_mm"],
    penguin_clean["Body_Mass_g"],
    alpha=0.6,
    c="steelblue"
)
ax1.set_title("Flipper Length vs Body Mass")
ax1.set_xlabel("Flipper Length (mm)")
ax1.set_ylabel("Body Mass (g)")

# Second scatter plot: Culmen Length vs Body Mass
ax2.scatter(
    penguin_clean["Culmen_Length_mm"],
    penguin_clean["Body_Mass_g"],
    alpha=0.6,
    c="darkorange"
)
ax2.set_title("Culmen Length vs Body Mass")
ax2.set_xlabel("Culmen Length (mm)")
ax2.set_ylabel("Body Mass (g)")

plt.tight_layout()
plt.show()


# In[38]:


avg_mass_sex = (
    penguin_clean
      .groupby(["Species", "Sex"])["Body_Mass_g"]
      .mean()
      .reset_index()
)
# Pivot so Sex becomes columns and Species rows
pivoted = avg_mass_sex.pivot(index="Species", columns="Sex", values="Body_Mass_g")


# In[39]:


ax = pivoted.plot(kind="barh", figsize=(8,5))  # <-- barh for horizontal

ax.set_title("Average Penguin Body Mass by Species and Sex")
ax.set_xlabel("Body Mass (g)")
ax.set_ylabel("Species")

plt.tight_layout()
plt.show()


# In[40]:


penguin_clean.to_csv("penguin_clean.csv", index=False)


# In[ ]:




