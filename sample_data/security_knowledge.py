"""
sample_data/security_knowledge.py — Curated Local Security Knowledge Base.

Contains verified, concise security reference documents covering common
Python vulnerabilities, CWE classifications, vulnerable patterns, and secure remediations.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Curated dataset of core Python security vulnerabilities
SECURITY_KNOWLEDGE_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "SEC-001-SQLI",
        "title": "SQL Injection (SQLi) via Dynamic Query Concatenation",
        "cwe": "CWE-89",
        "severity": "CRITICAL",
        "category": "injection",
        "keywords": ["sql", "sqlite", "postgres", "cursor.execute", "select", "insert", "format string", "f-string"],
        "content": (
            "SQL Injection (CWE-89): Occurs when untrusted user input is directly concatenated, "
            "formatted via f-strings, or interpolated into dynamic SQL queries instead of using "
            "parameterized queries or prepared statements. Attackers can alter query logic, bypass authentication, "
            "read sensitive tables, or drop databases.\n"
            "Vulnerable Pattern:\n"
            "    query = f\"SELECT * FROM users WHERE username = '{user_input}'\"\n"
            "    cursor.execute(query)\n"
            "Remediation:\n"
            "    Always use parameterized queries with query placeholders:\n"
            "    cursor.execute(\"SELECT * FROM users WHERE username = %s\", (user_input,))\n"
            "    Or use an ORM with parameterized filter expressions."
        ),
    },
    {
        "id": "SEC-002-CMD-INJ",
        "title": "OS Command Injection via Shell Execution",
        "cwe": "CWE-78",
        "severity": "CRITICAL",
        "category": "command_injection",
        "keywords": ["os.system", "subprocess.Popen", "subprocess.call", "shell=True", "exec", "eval", "spawn"],
        "content": (
            "OS Command Injection (CWE-78): Occurs when untrusted input is passed to system shells "
            "using os.system, os.popen, or subprocess.Popen/subprocess.run with shell=True. "
            "Attackers can inject shell metacharacters (;, &, |, `, $()) to execute arbitrary operating system commands.\n"
            "Vulnerable Pattern:\n"
            "    import os\n"
            "    os.system(f\"ping -c 1 {host}\")\n"
            "    subprocess.run(f\"backup.sh {user_file}\", shell=True)\n"
            "Remediation:\n"
            "    1. Avoid shell execution; set shell=False (default).\n"
            "    2. Pass arguments as a list of strings: subprocess.run([\"ping\", \"-c\", \"1\", host], shell=False, check=True)\n"
            "    3. Validate and whitelist argument formats."
        ),
    },
    {
        "id": "SEC-003-SSRF",
        "title": "Server-Side Request Forgery (SSRF)",
        "cwe": "CWE-918",
        "severity": "HIGH",
        "category": "network",
        "keywords": ["requests.get", "requests.post", "urllib.request", "httpx", "webhook", "fetch_url", "ssrf"],
        "content": (
            "Server-Side Request Forgery (CWE-918): Occurs when an application accepts a user-controlled URL "
            "and makes an outgoing network request to that URL without validating the destination host or IP address. "
            "Attackers can target internal microservices, cloud metadata APIs (e.g., http://169.254.169.254/), "
            "loopback addresses (localhost/127.0.0.1), or private subnet resources.\n"
            "Vulnerable Pattern:\n"
            "    url = request.args.get('url')\n"
            "    response = requests.get(url)\n"
            "Remediation:\n"
            "    1. Validate scheme (allow only http/https).\n"
            "    2. Maintain strict domain allowlists.\n"
            "    3. Resolve hostname and reject private/reserved IP ranges (RFC 1918, link-local, loopback)."
        ),
    },
    {
        "id": "SEC-004-PATH-TRAVERSAL",
        "title": "Path Traversal & Arbitrary File Access",
        "cwe": "CWE-22",
        "severity": "HIGH",
        "category": "file_access",
        "keywords": ["open", "os.path.join", "read_file", "send_file", "filename", "filepath", "..", "directory traversal"],
        "content": (
            "Path Traversal (CWE-22): Occurs when user-supplied filenames or relative paths containing '../' sequences "
            "or absolute paths are concatenated with base directories without canonicalization, permitting unauthorized "
            "reading or writing of arbitrary files on the filesystem (e.g., /etc/passwd, .env, application source).\n"
            "Vulnerable Pattern:\n"
            "    path = os.path.join(\"/var/data/uploads\", filename)\n"
            "    with open(path, 'r') as f: return f.read()\n"
            "Remediation:\n"
            "    1. Use os.path.basename() to strip directory components.\n"
            "    2. Canonicalize path and verify it stays inside intended base directory:\n"
            "       target = os.path.abspath(os.path.join(BASE_DIR, filename))\n"
            "       if not target.startswith(os.path.abspath(BASE_DIR)):\n"
            "           raise PermissionError('Path traversal detected')"
        ),
    },
    {
        "id": "SEC-005-DESERIALIZATION",
        "title": "Insecure Object Deserialization via Pickle or PyYAML",
        "cwe": "CWE-502",
        "severity": "CRITICAL",
        "category": "deserialization",
        "keywords": ["pickle.loads", "pickle.load", "_pickle", "yaml.load", "shelve", "unserialize"],
        "content": (
            "Insecure Deserialization (CWE-502): The Python pickle module and unconstrained yaml.load() "
            "can instantiate arbitrary Python classes and execute code defined in __reduce__ methods upon deserializing "
            "untrusted bytes, leading to immediate Remote Code Execution (RCE).\n"
            "Vulnerable Pattern:\n"
            "    import pickle\n"
            "    data = pickle.loads(user_provided_bytes)\n"
            "    import yaml\n"
            "    config = yaml.load(raw_yaml_string)  # UnsafeLoader\n"
            "Remediation:\n"
            "    1. Never unpickle untrusted data. Use safe formats like JSON, MessagePack, or Protocol Buffers.\n"
            "    2. For YAML, always use yaml.safe_load(data)."
        ),
    },
    {
        "id": "SEC-006-XXE",
        "title": "XML External Entity (XXE) Injection",
        "cwe": "CWE-611",
        "severity": "HIGH",
        "category": "xml_injection",
        "keywords": ["xml.etree", "lxml.etree", "parseString", "XMLParser", "xml", "ENTITY"],
        "content": (
            "XML External Entity Injection (CWE-611): Occurs when an XML parser processes XML input containing "
            "external entity references without disabling external DTD evaluation. Attackers can read local files, "
            "perform denial of service (Billion Laughs), or execute SSRF attacks.\n"
            "Vulnerable Pattern:\n"
            "    from lxml import etree\n"
            "    tree = etree.fromstring(untrusted_xml)\n"
            "Remediation:\n"
            "    1. Disable external entity resolution and DTD processing:\n"
            "       parser = etree.XMLParser(resolve_entities=False, no_network=True)\n"
            "    2. Use defusedxml package as a drop-in safe replacement for standard XML parsers."
        ),
    },
    {
        "id": "SEC-007-INSECURE-CRYPTO",
        "title": "Use of Broken or Weak Cryptographic Algorithms",
        "cwe": "CWE-327",
        "severity": "MEDIUM",
        "category": "cryptography",
        "keywords": ["hashlib.md5", "hashlib.sha1", "DES", "RC4", "ECB", "random.random", "random.randint"],
        "content": (
            "Weak Cryptography (CWE-327 & CWE-338): Using obsolete hash algorithms (MD5, SHA1) for security-sensitive "
            "operations or passwords, or using the standard pseudo-random number generator (random module) for security tokens, "
            "session IDs, or password reset tokens. Standard random is predictable via Mersenne Twister state recovery.\n"
            "Vulnerable Pattern:\n"
            "    import hashlib, random\n"
            "    token = str(random.random())\n"
            "    password_hash = hashlib.md5(password.encode()).hexdigest()\n"
            "Remediation:\n"
            "    1. For passwords, use argon2-cffi, bcrypt, or hashlib.pbkdf2_hmac with salt.\n"
            "    2. For cryptographic tokens, use secrets.token_urlsafe() or secrets.token_hex().\n"
            "    3. For data hashing, use SHA-256 or SHA-3."
        ),
    },
    {
        "id": "SEC-008-CORS-CSRF",
        "title": "Missing CSRF Protection & Overly Permissive CORS",
        "cwe": "CWE-352",
        "severity": "MEDIUM",
        "category": "web_security",
        "keywords": ["Access-Control-Allow-Origin", "CORS", "*", "csrf", "cookies", "samesite"],
        "content": (
            "Cross-Site Request Forgery (CWE-352) & Wildcard CORS (CWE-942): State-changing HTTP endpoints "
            "(POST/PUT/DELETE) that accept cookie authentication without anti-CSRF tokens or SameSite cookie flags "
            "can be triggered by third-party origins on behalf of authenticated users. Similarly, returning "
            "Access-Control-Allow-Origin: * with credentials enabled exposes sensitive API endpoints.\n"
            "Vulnerable Pattern:\n"
            "    @app.route('/transfer', methods=['POST'])\n"
            "    def transfer():\n"
            "        # No CSRF token verified\n"
            "        execute_wire(request.form['amount'])\n"
            "Remediation:\n"
            "    1. Require CSRF tokens on all mutating requests (e.g. Flask-WTF, Django CSRF middleware).\n"
            "    2. Set Set-Cookie with SameSite=Lax or SameSite=Strict and Secure.\n"
            "    3. Specify explicit trusted origins in CORS headers."
        ),
    },
]
