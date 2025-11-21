#!/usr/bin/env python3
"""Utility to generate bcrypt password hashes for use in .env.
Usage:
    python tools/hash_password.py mypassword
Outputs the bcrypt hash to stdout.
"""
import sys

try:
    import bcrypt
except Exception as e:
    print('bcrypt not installed. Install with: pip install bcrypt')
    sys.exit(2)

if len(sys.argv) < 2:
    print('Usage: python tools/hash_password.py <password>')
    sys.exit(1)

pwd = sys.argv[1].encode('utf-8')
h = bcrypt.hashpw(pwd, bcrypt.gensalt())
print(h.decode('utf-8'))

