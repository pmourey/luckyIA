#!/usr/bin/env python3
"""Set ADMIN_PASSWORD in .env using bcrypt hash of a provided plaintext password.

Usage:
  python tools/set_admin_password.py mysecret      # prints bcrypt hash (does not modify .env)
  python tools/set_admin_password.py --set mysecret  # write ADMIN_PASSWORD to project-root/.env (won't overwrite unless --force)
  python tools/set_admin_password.py --set --force mysecret  # force overwrite
  python tools/set_admin_password.py --env-path /path/to/.env --set mysecret

Note: requires `bcrypt` package. Install with `pip install bcrypt`.
"""
import argparse
import os
import sys
import bcrypt
import re


def hash_password(plaintext: str, rounds: int = 12) -> str:
    pw = plaintext.encode('utf-8')
    salt = bcrypt.gensalt(rounds)
    return bcrypt.hashpw(pw, salt).decode('utf-8')


def env_path_default() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    return os.path.join(project_root, '.env')


def read_env(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ''


def write_env(path: str, key: str, value: str, force: bool = False) -> None:
    content = read_env(path)
    pattern = re.compile(rf'^(%s)\s*=.*$' % re.escape(key), flags=re.MULTILINE)
    if pattern.search(content):
        if not force:
            raise RuntimeError(f"Key {key} already exists in {path}. Use --force to overwrite.")
        new_content = pattern.sub(f"{key}={value}", content)
    else:
        if content and not content.endswith('\n'):
            content += '\n'
        new_content = content + f"{key}={value}\n"
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(new_content)
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(description='Hash a password and optionally set ADMIN_PASSWORD in .env')
    p.add_argument('password', nargs=1, help='Plaintext password to hash and set')
    p.add_argument('--set', action='store_true', help='Write ADMIN_PASSWORD to .env')
    p.add_argument('--force', action='store_true', help='When used with --set, overwrite existing ADMIN_PASSWORD')
    p.add_argument('--env-path', default=None, help='Path to .env file (default: project-root/.env)')
    p.add_argument('--rounds', type=int, default=12, help='bcrypt rounds/work factor (default 12)')
    args = p.parse_args(argv)

    plaintext = args.password[0]
    if not plaintext:
        print('Empty password not allowed', file=sys.stderr)
        return 2

    try:
        h = hash_password(plaintext, rounds=args.rounds)
    except Exception as e:
        print('Error hashing password:', e, file=sys.stderr)
        return 2

    if args.set:
        env_path = args.env_path or env_path_default()
        try:
            write_env(env_path, 'ADMIN_PASSWORD', h, force=args.force)
            print(f'Wrote ADMIN_PASSWORD to {env_path}')
            print(h)
            return 0
        except Exception as e:
            print(f'Failed to write .env: {e}', file=sys.stderr)
            return 2
    else:
        # print hash only
        print(h)
        return 0


if __name__ == '__main__':
    sys.exit(main())

