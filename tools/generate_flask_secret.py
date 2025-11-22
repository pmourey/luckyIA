#!/usr/bin/env python3
"""Generate a secure FLASK_SECRET and optionally write it to the project's .env file.

Usage:
  python tools/generate_flask_secret.py         # print a new secret to stdout
  python tools/generate_flask_secret.py --set   # write to .env (won't overwrite existing key unless --force)
  python tools/generate_flask_secret.py --set --force  # force overwrite existing FLASK_SECRET in .env
  python tools/generate_flask_secret.py --length 48  # use 48 bytes of randomness

This script is safe by default: it prints the secret and does not modify .env unless --set is provided.
"""
import argparse
import secrets
import os
import sys
import re


def generate_secret(nbytes: int) -> str:
    return secrets.token_urlsafe(nbytes)


def env_path_default() -> str:
    # Default .env located at project root (one level up from tools/)
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
    """Write or update a single KEY=VALUE pair in a .env file.
    If the key exists and force is False, raises RuntimeError.
    """
    content = read_env(path)
    pattern = re.compile(rf'^(%s)\s*=.*$' % re.escape(key), flags=re.MULTILINE)
    if pattern.search(content):
        if not force:
            raise RuntimeError(f"Key {key} already exists in {path}. Use --force to overwrite.")
        # replace existing line(s)
        new_content = pattern.sub(f"{key}={value}", content)
    else:
        # append
        if content and not content.endswith('\n'):
            content += '\n'
        new_content = content + f"{key}={value}\n"
    # write atomically
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(new_content)
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate FLASK_SECRET and optionally write to .env')
    p.add_argument('--set', action='store_true', help='Write the generated secret to the .env file (default: do not modify files)')
    p.add_argument('--force', action='store_true', help='When used with --set, overwrite existing FLASK_SECRET in .env')
    p.add_argument('--env-path', default=None, help='Path to .env file (default: project-root/.env)')
    p.add_argument('--length', type=int, default=32, help='Number of random bytes for secret (passed to token_urlsafe). Default 32')
    p.add_argument('--print-only', action='store_true', help='Print the secret only (same as no --set).')
    args = p.parse_args(argv)

    secret = generate_secret(args.length)

    if args.set and not args.print_only:
        env_path = args.env_path or env_path_default()
        try:
            write_env(env_path, 'FLASK_SECRET', secret, force=args.force)
            print(f"Wrote FLASK_SECRET to {env_path}")
            print(secret)
        except Exception as e:
            print(f"Error writing .env: {e}", file=sys.stderr)
            return 2
    else:
        # default behavior: print only
        print(secret)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

