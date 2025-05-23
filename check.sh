#!/bin/bash

# 设置根目录
DB_DIR="./db"

echo "🔍 Checking DB files recursively in: $DB_DIR"
echo

# 查找常见的数据库文件后缀
find "$DB_DIR" -type f \( -name "*.db" -o -name "*.sqlite" -o -name "*.sqlite3" \) | while read db_file; do
    if [ -f "$db_file" ]; then
        perms=$(ls -l "$db_file")
        if [ -w "$db_file" ]; then
            status="✅ Writable"
        else
            status="❌ Not Writable"
        fi
        echo "$perms --> $status"
    fi
done
