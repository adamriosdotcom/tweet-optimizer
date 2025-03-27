#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 20:50:32 2025

@author: adamrios
"""

import sqlite3
import pandas as pd

conn = sqlite3.connect('tweets.db')

# Check rows with non-empty topics
query = "SELECT COUNT(*) FROM tweets WHERE topics IS NOT NULL AND TRIM(topics) != ''"
filled_count = pd.read_sql_query(query, conn).iloc[0, 0]

# Total rows count
total_query = "SELECT COUNT(*) FROM tweets"
total_count = pd.read_sql_query(total_query, conn).iloc[0, 0]

conn.close()

print(f"Rows with topics: {filled_count}")
print(f"Total rows: {total_count}")
print(f"Rows without topics: {total_count - filled_count}")