# InstamartCapstone

## 1. Overview

InstamartCapstone is a data analysis and machine learning project using generated Instamart-style customer, product, order, and support-ticket data.

The project is divided into four parts:

- **Part A:** Data audit, cleaning, database storage, SQL analysis, and Pandas analysis.
- **Part B:** Customer segmentation and lapse prediction.
- **Part C:** Support-ticket triage.
- **Part D:** Agent workflows and refund processing.

## 2. Reproduce

Run the following commands from the project root directory.

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate the environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Generate the project data:

   ```powershell
   python generate_data.py
   ```

5. Audit the raw data:

   ```powershell
   python src/part_a/01_data_audit.py
   ```

6. Clean the data:

   ```powershell
   python src/part_a/02_clean_data.py
   ```

7. Store the cleaned data:

   ```powershell
   python src/part_a/03_store_data.py
   ```

8. Validate the cleaned data and database:

   ```powershell
   python src/part_a/04_validate_data.py
   ```

9. Run the SQL analysis:

   ```powershell
   python src/part_a/05_sql_analysis.py
   ```

10. Run the Pandas analysis:

    ```powershell
    python src/part_a/06_pandas_analysis.py
    ```

## 3. Part A

### Data audit

The raw datasets contain:

| Dataset | Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 13,084 |
| Tickets | 164 |

Column	Data type	Missing values	Missing %

Column	Data type	Missing values	Missing %
customer_id	str	0	0.00%
city	str	0	0.00%
signup_date	str	0	0.00%
age	float64	61	5.08%
gender	str	30	2.50%

Column	Data type	Missing values	Missing %
product_id	str	0	0.00%
product_name	str	0	0.00%
category	str	0	0.00%
price_inr	int64	0	0.00%

Column	Data type	Missing values	Missing %
order_id	str	0	0.00%
customer_id	str	0	0.00%
product_id	str	0	0.00%
quantity	int64	0	0.00%
order_ts	str	0	0.00%
status	str	0	0.00%
discount_pct	int64	0	0.00%
order_amount	str	53	0.41%
delivery_minutes	float64	2,541	19.42%

Important audit findings:

- Raw orders contained 119 exact duplicate rows.
- There were 656 pending orders.
- There were 12,309 distinct order IDs.
- Order amounts contained 120 nonnumeric values.
- Delivery times contained 106 negative values and 2,541 blank values.
- Customer data contained 61 blank ages and 6 ages outside the 10–100 range.
- Raw customer data contained 24 city spellings, 10 distinct nonblank gender spellings, and 30 blank gender values.

### Data cleaning

Customer cleaning results:

- Standardized city names into six cities.
- Standardized gender values into female, male, and unknown.
- Handled missing and invalid ages using city-wise median imputation.
- Converted signup dates to datetime format.

Final customer counts:

| City | Customers |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

| Gender | Customers |
|---|---:|
| Female | 610 |
| Male | 560 |
| Unknown | 30 |

Order cleaning results:

| Status | Orders |
|---|---:|
| Completed | 11,110 |
| Cancelled | 1,199 |
| Pending | 0 |
| **Total** | **12,309** |

**Order cleaning rules**

The following rules were applied in the specified order:

1. **Datetime conversion:** Converted `order_ts` into datetime format.
2. **Exact duplicate removal:** Removed 119 exact duplicate rows.
3. **Duplicate order ID handling:** Kept the latest record for each `order_id`, removing 656 older records.
4. **Order amount correction:** Converted `order_amount` to numeric, treated invalid values as missing, and repaired 112 missing amounts using the product price, quantity and discount percentage.
5. **Delivery-time correction:** Replaced negative delivery times with missing values (105 values corrected). Missing delivery times for completed orders were filled using the median delivery time of the corresponding product category. Cancelled orders retained missing delivery times.

**Final order-cleaning results**

| Measure | Result |
|---|---:|
| Total cleaned orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Pending orders | 0 |
| Missing order amounts | 0 |
| Missing delivery times in completed orders | 0 |
| Missing delivery times in cancelled orders | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |


### Database validation

Cleaned customer, product, and order data were stored in `data/instamart.db`.

The validation script checked 10 reference checkpoints and printed:

`ALL CHECKPOINTS PASSED`

### SQL analysis

Seven SQL queries were executed.

| Query | Result |
|---|---:|
| Q1: Customers with qualifying completed-order spend | 30 |
| Q2: City and category revenue analysis | 30 rows |
| Q3: Products priced above average | 11 |
| Q4: Customers without completed orders | 25 |
| Q5: Customers spending above average | 550 |
| Q6: Ranking analysis | 1,175 rows |
| Q7: Daily revenue running total | 183 days |

Total completed-order revenue: **INR 2,998,668.80**.
The average completed-order spend per customer was INR 2,552.06.

The highest city/category revenue combination was Bengaluru Electronics, at INR 233,909.45.

For Q6, RANK and ROW_NUMBER can differ when customers have tied spending values: RANK assigns equal ranks to ties, while ROW_NUMBER assigns a unique row number.

| City | Customer ID | Spend |
|---|---|---:|
| Bengaluru | C0082 | INR 12,856.20 |
| Chennai | C0930 | INR 10,151.40 |
| Delhi | C0291 | INR 13,742.60 |
| Hyderabad | C0743 | INR 12,730.05 |
| Mumbai | C0693 | INR 14,766.55 |
| Pune | C0426 | INR 10,377.10 |


### Pandas analysis

The Pandas pivot-table revenue total matched the SQL revenue total: INR 2,998,668.80.

| City | Total revenue | Average order amount | Completed orders |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Chennai | INR 463,629.15 | INR 272.72 | 1,700 |
| Delhi | INR 502,829.80 | INR 243.62 | 2,064 |
| Hyderabad | INR 528,589.90 | INR 284.95 | 1,855 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |
| Pune | INR 500,918.05 | INR 252.48 | 1,984 |


Completed-order statistics:

| Measure | Result |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | 4.61 |
| First quartile (Q1) | INR 64.80 |
| Third quartile (Q3) | INR 297.00 |
| Upper IQR fence | INR 645.30 |
| Outliers above upper fence | 897 |
| Lower IQR fence | INR -283.50 |
| Outliers below lower fence | 0 |

The strongest absolute correlation was between `order_amount` and `discount_pct`, at -0.095. This is a very weak negative correlation. Correlation does not establish causation.

### Interpretation

The completed orders generated INR 2,998,668.80 in revenue. Hyderabad generated the highest city-wise revenue, followed by Bengaluru. Mumbai had the highest average order amount.

The order amount distribution was positively skewed, with a mean of INR 269.91 and a median of INR 135. The IQR method identified 897 outliers above the upper fence.

The strongest correlation was between order amount and discount percentage, but its value of -0.095 indicates a very weak negative relationship. The other analysed correlations were also close to zero.

## 4. Part B - Customer Segmentation and Lapse Prediction


### Task 1 - RFM Analysis

RFM (Recency, Frequency, Monetary) analysis was performed to understand customer purchasing behaviour using completed orders.

The analysis used a snapshot date of **2025-10-01**.

The following three metrics were calculated for each customer:

- **Recency:** Number of days since the customer's most recent completed order.
- **Frequency:** Number of distinct completed orders placed by the customer.
- **Monetary:** Total amount spent by the customer on completed orders.

Only customers with at least one completed order were included.

**Key results:**

- Total customers analysed: 1,175
- Average frequency: 9.46 orders
- Median recency: 19 days
- Maximum frequency: 33 orders
- Average monetary value: ₹2,552.06
- Maximum monetary value: ₹14,766.55

**Output:** `outputs/customer_rfm.csv`

**Python script:** `src/part_b/01_build_rfm.py`

### Task 2 - Customer Clustering

K-Means clustering was applied to the RFM features (Recency, Frequency, and Monetary) to group customers based on their purchasing behaviour.

Before clustering, the RFM features were standardized using `StandardScaler`.

The following K values were evaluated:

- K = 2 to K = 8
- K-Means parameters: `n_init=10`, `random_state=42`
- Evaluation metrics: Inertia and Silhouette Score

**Clustering evaluation results:**

| K | Inertia | Silhouette Score |
|---|---:|---:|
| 2 | 1820.761 | 0.465 |
| 3 | 1268.235 | 0.445 |
| 4 | 743.071 | 0.526 |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

**Selected number of clusters: K = 4**

K = 4 was selected because it achieved the highest Silhouette Score among the evaluated values, provided a reasonable elbow in the inertia curve, and produced interpretable customer groups.

**Output:** `outputs/clustering_evaluation.csv`

**Python script:** `src/part_b/02_customer_clustering.py`

### Task 3 - Customer Segmentation

The four customer clusters generated in Task 2 were analysed using their average Recency, Frequency, and Monetary values.

Each cluster was assigned a meaningful business label based on its purchasing behaviour.

**Customer segment profiles:**

| Segment ID | Segment Name | Customers | Avg. Recency | Avg. Frequency | Avg. Monetary |
|---|---|---:|---:|---:|---:|
| 0 | Regular Low-Spend | 436 | 28.86 | 3.77 | ₹1,057.62 |
| 1 | Premium | 138 | 24.12 | 6.77 | ₹7,129.70 |
| 2 | Frequent Active | 419 | 8.77 | 19.40 | ₹3,410.02 |
| 3 | At-Risk | 182 | 109.04 | 2.20 | ₹685.99 |

**Segment interpretation:**

- **Regular Low-Spend:** Customers who purchase occasionally and spend relatively less.
- **Premium:** Customers with the highest average monetary value.
- **Frequent Active:** Customers who purchase frequently and have placed recent orders.
- **At-Risk:** Customers with high recency values and low purchase frequency.

The Premium segment was identified using the highest average monetary value.

**PCA analysis:**

Principal Component Analysis (PCA) was used to reduce the standardized RFM data to two dimensions for visualization.

- First principal component: 63.43% variance explained.
- Second principal component: 21.80% variance explained.
- Total variance explained: 85.23%.

**Outputs:**

- `outputs/segment_profile.csv`
- `outputs/customer_pca.csv`
- `outputs/customer_segments.csv`

**Python script:** `src/part_b/03_profile_segments.py`

### Task 4 - Lapse Dataset

A customer lapse prediction dataset was created to identify customers who stopped placing completed orders.

**Lapse definition:**

- Cutoff date: `2025-09-01`
- Observation period: 1 September 2025 to 30 September 2025.
- Customers with at least one completed order before the cutoff date were included.
- `lapsed = 1`: The customer placed no completed orders during September 2025.
- `lapsed = 0`: The customer placed at least one completed order during September 2025.

**Features used:**

- Recency
- Frequency
- Monetary
- Average discount percentage
- Cancellation rate
- Average delivery time
- Favourite product category
- City
- Age
- Gender
- Customer tenure in days

Only information available before the cutoff date was used to create the prediction features. September order activity was used only to determine the target label.

**Dataset summary:**

- Total customers: 1,156
- Lapsed customers: 443
- Not-lapsed customers: 713
- Lapse rate: 38.32%
- Training records: 924
- Testing records: 232

**Output:** `outputs/lapse_dataset.csv`

**Python script:** `src/part_b/04_build_lapse_dataset.py`

### Task 5 - Data Preprocessing

The lapse prediction dataset was preprocessed before training the classification models.

**Feature selection:**

- `customer_id` was excluded from the model input.
- `lapsed` was used as the target variable.
- Numerical and categorical features were handled separately.

**Numerical preprocessing:**

The following numerical features were standardized using `StandardScaler`:

- Recency
- Frequency
- Monetary
- Average discount percentage
- Cancellation rate
- Average delivery time
- Age
- Customer tenure in days

**Categorical preprocessing:**

The following categorical features were encoded using `OneHotEncoder` with `drop="first"` and `handle_unknown="ignore"`:

- Favourite product category
- City
- Gender

**Train-test split:**

- Training data: 924 records
- Testing data: 232 records
- Test size: 20%
- Stratification: Enabled
- Random state: 42

To prevent data leakage, the scaler and encoder were fitted only on the training data. The fitted transformations were then applied to the test data.

**Processed feature dimensions:**

- Training data: 924 rows × 19 features
- Testing data: 232 rows × 19 features

No missing values were present in the processed training or testing data.

**Python script:** `src/part_b/05_preprocess_lapse.py`

### Task 6 - Model Training and Evaluation

Two classification models were trained to predict customer lapse:

- **Logistic Regression:** Hyperparameter tuning was performed using GridSearchCV with 5-fold cross-validation. The best parameter was `C=0.1`.
- **Random Forest:** Hyperparameter tuning was performed using GridSearchCV with 5-fold cross-validation. The best parameters were `max_depth=3` and `min_samples_leaf=5`.

To prevent data leakage during cross-validation, preprocessing was included inside each model's Scikit-learn Pipeline. This ensured that the scaler and encoder were fitted separately within each cross-validation training fold.

**Baseline performance:**
- Accuracy: 61.64%
- Precision: 0
- Recall: 0
- F1-score: 0

**Test-set model performance:**

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 76.29% | 76.72% |
| Precision | 65.45% | 68.04% |
| Recall | 80.90% | 74.16% |
| F1-score | 72.36% | 70.97% |

**Model selection:**

Logistic Regression was selected because it achieved a higher F1-score and recall than Random Forest. This makes it more suitable for identifying potentially lapsed customers for retention campaigns.

### Task 7 - Oversampling

The training dataset had an imbalanced target distribution:

- Not-lapsed customers: 570
- Lapsed customers: 354

To address this class imbalance, random oversampling was applied to the training data.

**Oversampling procedure:**

- All original majority-class records were retained.
- All original minority-class records were retained.
- Additional lapsed-customer records were sampled with replacement until both classes had equal counts.
- Oversampling was performed only on the training data. The test data was not modified.

**Balanced training distribution:**

| Class | Records |
|---|---:|
| Not-lapsed | 570 |
| Lapsed | 570 |
| **Total** | **1,140** |

The Logistic Regression model was trained again using the balanced training data.

**Performance after oversampling:**

- Precision: 58.27%
- Recall: 91.01%
- F1-score: 71.05%

Oversampling improved recall, helping the model identify more lapsed customers. However, precision decreased, meaning more non-lapsed customers could be incorrectly flagged.

This approach is useful when the retention team prioritizes identifying as many potentially lapsed customers as possible.

### Task 8 - Model Interpretation and Retention List

This task focused on understanding model predictions and identifying Premium customers who may need retention campaigns.

**1. Random Forest feature importance**

The five most important features were:

| Feature | Importance |
|---|---:|
| Frequency | 0.352171 |
| Monetary | 0.288671 |
| Recency | 0.114974 |
| Cancellation Rate | 0.053057 |
| Average Discount Percentage | 0.040657 |

Frequency and monetary value were the two most influential features in the Random Forest model.

**2. Logistic Regression coefficients**

The three highest-ranked coefficients by absolute magnitude were:

| Feature | Coefficient |
|---|---:|
| Frequency | -1.281146 |
| Favourite Category: Electronics | -0.378188 |
| Favourite Category: Snacks | -0.202604 |

The negative coefficients indicate an association with lower predicted odds of lapse, holding other model inputs constant. For categorical features, the interpretation is relative to the reference category.

**3. Balanced Logistic Regression model**

The Logistic Regression model was trained again using the balanced training dataset from Task 7.

- Training records after oversampling: 1,140
- Not-lapsed customers: 570
- Lapsed customers: 570

The trained model was saved as `outputs/lapse_model.joblib` and reloaded. Predictions from the reloaded model matched the original model predictions.

**4. Premium customer retention list**

The model predicted lapse probabilities for customers in the test dataset. These predictions were used to identify the top 10 Premium customers with the highest lapse probabilities.

The retention list was generated using Premium customers present in the 232-record test dataset. It does not represent all 138 Premium customers in the complete dataset.

**Output:** `outputs/retention_list.csv`

The retention list contains:

- Customer ID
- Predicted lapse probability
- Actual lapse status

Among the top 10 Premium customers, one customer was actually lapsed.

**Important limitation:** Premium customer segments were created using full-period RFM data, including September 2025. Therefore, this retention list is an illustrative post-hoc analysis and not a leakage-free evaluation of Premium customer retention performance. The lapse prediction model itself uses pre-cutoff features.

**Python script:** `src/part_b/08_model_interpretation.py`

# Part C — Intelligent Support Ticket Classification and Routing

## 1. Objective

Part C implements an intelligent customer support ticket processing pipeline.

The system audits incoming tickets, masks personally identifiable information (PII), classifies ticket categories, demonstrates token-budget management, routes tickets to appropriate support teams, and evaluates classification performance.

## 2. Project Structure

```text
src/
└── part_c/
    ├── 01_ticket_audit.py
    ├── 02_mask_pii.py
    ├── 03_ticket_classification.py
    ├── 04_token_budget_demo.py
    ├── 05_final_triage.py
    └── 06_confusion_matrix.py
```

## 3. Script Descriptions

| Script | Description |
|---|---|
| 01_ticket_audit.py | Audits support tickets, checks empty and long tickets, and analyzes category distribution. |
| 02_mask_pii.py | Handles masking of personally identifiable information. |
| 03_ticket_classification.py | Implements ticket classification using zero-shot and few-shot approaches. |
| 04_token_budget_demo.py | Demonstrates token-budget management. |
| 05_final_triage.py | Performs final ticket classification, priority assignment, and routing. |
| 06_confusion_matrix.py | Evaluates classification performance using a confusion matrix. |

## 4. Classification Results

| Method | Correct classifications | Accuracy |
|---|---:|---:|
| Zero-shot | 126/147 | 85.71% |
| Few-shot | 147/147 | 100% |

Few-shot classification achieved 100% accuracy on the evaluated 147 tickets.

## 5. Ticket Routing Results

A total of 164 support tickets were processed.

| Route | Tickets | Purpose |
|---|---:|---|
| support_queue | 102 | Standard customer support |
| security_team | 34 | Fraud-related cases |
| human_review | 17 | Unclassified or problematic tickets |
| premium_desk | 11 | Premium customer cases |
| **Total** | **164** | |

## 6. Priority Distribution

| Priority | Tickets |
|---|---:|
| NORMAL | 119 |
| HIGH | 45 |
| **Total** | **164** |

## 7. Routing Validation

The following checks passed:

- All 164 tickets were accounted for.
- All 9 invalid-output tickets were routed to human review.
- All 4 long tickets were routed to human review.
- All 4 empty tickets were routed to human review.
- Fraud-related tickets were routed to the security team.
- Premium customer tickets were routed to the premium desk.

## 8. Execution Commands

Run the scripts from the project root directory with the virtual environment activated.

```powershell
python src/part_c/01_ticket_audit.py
python src/part_c/02_mask_pii.py
python src/part_c/03_ticket_classification.py
python src/part_c/04_token_budget_demo.py
python src/part_c/05_final_triage.py
python src/part_c/06_confusion_matrix.py
```

## 9. Final Outcome

Part C successfully demonstrates an intelligent support-ticket processing workflow involving ticket auditing, PII handling, classification, token-budget management, ticket routing, and model evaluation.

The verified results include:

- Few-shot classification accuracy: 100% (147/147).
- Total tickets routed: 164/164.
- Human-review tickets: 17.
- Priority assignment: 119 NORMAL and 45 HIGH.

**Part C implementation and routing verification completed.**

# Part D — Policy-Based Refund Automation Agent

## 1. Objective

Part D implements an intelligent refund automation system using Retrieval-Augmented Generation (RAG), policy-based decision-making, refund tools, and a ReAct agent.

The system retrieves relevant refund policies, evaluates customer refund requests, executes appropriate actions, and provides human handoff when necessary.

## 2. Project Structure

```text
src/
└── part_d/
    ├── 01_policy_chunking.py
    ├── 02_policy_retrieval.py
    ├── 03_policy_rag.py
    ├── 04_policy_evaluation.py
    ├── 05_refund_tools.py
    └── 06_react_refund_agent.py
```

## 3. Script Descriptions

| Script | Description |
|---|---|
| 01_policy_chunking.py | Splits refund policy documents into manageable text chunks. |
| 02_policy_retrieval.py | Retrieves relevant policy information for customer queries. |
| 03_policy_rag.py | Implements Retrieval-Augmented Generation for refund policy questions. |
| 04_policy_evaluation.py | Evaluates policy retrieval and answer quality using test questions. |
| 05_refund_tools.py | Implements refund calculation, eligibility checks, and refund-related operations. |
| 06_react_refund_agent.py | Implements the ReAct agent that reasons through customer requests and uses available tools. |

## 4. Refund Agent Workflow

The refund agent follows a ReAct-style workflow:

1. Receives a customer refund request.
2. Analyzes the request.
3. Selects the appropriate refund tool when necessary.
4. Retrieves relevant order and policy information.
5. Evaluates refund eligibility.
6. Produces a final decision or requests human intervention.

## 5. Safety Mechanisms

The refund agent implements safety limits to prevent uncontrolled execution.

| Safety mechanism | Configuration |
|---|---|
| Maximum token budget | 6,000 tokens |
| Maximum execution time | 15 seconds |
| Maximum iterations | Controlled by the agent's iteration limit |
| Token-limit fallback | FALLBACK_TOKEN_LIMIT |
| Time-limit fallback | FALLBACK_TIMEOUT |
| Iteration fallback | FALLBACK_MAX_ITERATIONS |

When a safety limit is exceeded, the agent returns a human handoff response.

## 6. Refund Agent Evaluation

Seven test scenarios were executed.

| Scenario | Expected outcome | Result |
|---|---|---|
| S1 | REFUND | PASS |
| S2 | REFUND | PASS |
| S3 | ESCALATE | PASS |
| S4 | DENIED | PASS |
| S5 | DENIED | PASS |
| S6 | NOT_FOUND | PASS |
| S7 | FALLBACK_MAX_ITERATIONS | PASS |

### Evaluation Summary

- Total scenarios: 7
- Passed: 7
- Failed: 0
- Success rate: 100%

Evaluation results are saved in:

`data/processed/agent_evaluation.jsonl`

## 7. Safety Limit Testing

Additional tests verified the agent's fallback mechanisms.

### Token-limit test

- Configured token limit: 0
- Actual tokens used: 25
- Result: FALLBACK_TOKEN_LIMIT
- Human handoff: Successful

### Time-limit test

- Configured time limit: 0 seconds
- Model calls: 0
- Result: FALLBACK_TIMEOUT
- Human handoff: Successful

Both safety-limit tests passed.

## 8. Execution Commands

Run the scripts from the project root directory with the virtual environment activated.

```powershell
python src/part_d/01_policy_chunking.py
python src/part_d/02_policy_retrieval.py
python src/part_d/03_policy_rag.py
python src/part_d/04_policy_evaluation.py
python src/part_d/05_refund_tools.py
python src/part_d/06_react_refund_agent.py
```

## 9. Final Outcome

Part D demonstrates a policy-based refund automation workflow using RAG, refund tools, and a ReAct agent.

The verified results include:

- Seven out of seven refund-agent scenarios passed.
- Token-limit fallback verified.
- Time-limit fallback verified.
- Maximum-iteration fallback verified.
- Human handoff behaviour verified.

**Part D refund-agent evaluation and safety-limit validation completed successfully.**

## 10. Limitations and Future Improvements

### Limitations

- The project uses synthetically generated data rather than live Instamart customer and transaction records.
- The classification and refund-agent workflows use a mock LLM implementation for offline, deterministic testing.
- Model predictions and evaluation results should be validated further using real-world production data before deployment.
- The Premium retention list is an illustrative, post-hoc analysis because Premium segments were created using full-period RFM data, including September. It is not a leakage-free evaluation.

### Data Leakage Prevention

- The scaler and encoder were fitted only on training data.
- Oversampling was performed only on training data.
- No test rows were used during these operations.
- No order data dated on or after 2025-09-01 was used to create lapse prediction features.
- Customer segment labels from Task 3 were not used as model input features.

### Future Improvements

- Integrate a live LLM API instead of the mock LLM implementation.
- Connect the workflow to real-time customer and order databases.
- Improve model evaluation using additional real-world test data.
- Add monitoring, logging, and production deployment capabilities.
- Improve human-review workflows for complex customer support and refund cases.

### Project Completion Status

- Part A — Data Processing and SQL: Completed.
- Part B — Customer Segmentation and Lapse Prediction: Completed.
- Part C — Intelligent Support Ticket Classification and Routing: Completed and verified.
- Part D — Policy-Based Refund Automation Agent: Completed and verified.

**All four project parts have been implemented, documented, and tested using the available evaluation scenarios.**