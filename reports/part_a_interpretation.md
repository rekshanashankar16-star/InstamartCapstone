# Part A - Data Analysis Interpretation

The completed orders generated total revenue of INR 2,998,668.80. Pandas revenue calculations matched the SQL analysis, confirming consistency.

The city-wise analysis shows that Hyderabad generated the highest revenue, followed by Bengaluru. Mumbai had the highest average order amount.

The order amount distribution is positively skewed, with a mean of INR 269.91 and median of INR 135. The analysis identified 897 outliers using the IQR method, indicating that a relatively small number of orders have unusually high values.

The strongest correlation was between `order_amount` and `discount_pct`, with a coefficient of -0.095. This represents a very weak negative relationship. The other correlations were also close to zero, suggesting no strong linear relationships among the analysed variables. Correlation does not imply causation.