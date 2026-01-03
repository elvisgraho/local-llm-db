"""
MITRE ATT&CK Knowledge Graph for transparent intent detection.

This module provides attack chain awareness and tactic relationships
to improve retrieval without requiring users to speak in MITRE terminology.

Research basis:
- MITRE ATT&CK Framework (https://attack.mitre.org)
- Cyber Kill Chain (Lockheed Martin)
- Multi-hop reasoning for attack chains
"""

from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class MITRETactic:
    """MITRE ATT&CK Tactic with relationships."""
    name: str
    tactic_id: str
    keywords: List[str]  # Natural language patterns
    tools: List[str]  # Common tools for this tactic
    requires: List[str]  # Prerequisites (other tactics often needed first)
    leads_to: List[str]  # Common next steps in attack chain
    techniques: List[str]  # Key technique IDs


# ============================================================================
# MITRE ATT&CK Tactic Definitions (Enterprise Matrix)
# ============================================================================

MITRE_TACTICS = {
    "Reconnaissance": MITRETactic(
        name="Reconnaissance",
        tactic_id="TA0043",
        keywords=[
            "recon", "osint", "reconnaissance", "footprint",
            "research", "gather intel", "target research",
            "dns enum", "subdomain enum", "email harvest",
            "port scan", "enumerate buckets", "whois",
            "github scraping", "shodan search", "dorking"
        ],
        tools=[
            "maltego", "recon-ng", "theharvester", "amass",
            "subfinder", "shodan", "censys", "dnsdumpster",
            "spiderfoot", "masscan", "nuclei", "phonebook.cz"
        ],
        requires=[],
        leads_to=["Resource Development", "Initial Access"],
        techniques=["T1595", "T1592", "T1589", "T1590", "T1598", "T1593"]
    ),

    "Resource Development": MITRETactic(
        name="Resource Development",
        tactic_id="TA0042",
        keywords=[
            "infrastructure", "buy domain", "vps", "server setup",
            "ssl certificate", "persona", "accounts", "botnet",
            "payload staging", "acquire", "lease", "host",
            "buy exploits", "cryptocurrency"
        ],
        tools=[
            "terraform", "digitalocean", "namecheap", "letsencrypt",
            "evilginx", "modlishka", "mythic", "cobalt strike"
        ],
        requires=["Reconnaissance"],
        leads_to=["Initial Access"],
        techniques=["T1583", "T1584", "T1585", "T1587", "T1588", "T1608"]
    ),

    "Initial Access": MITRETactic(
        name="Initial Access",
        tactic_id="TA0001",
        keywords=[
            "exploit", "vulnerability", "rce", "remote code execution",
            "phishing", "spearphishing", "breach", "compromise",
            "entry point", "foothold", "initial compromise",
            "public exploit", "0day", "exploit kit", "valid accounts",
            "drive-by", "supply chain", "password spray", "cloud console"
        ],
        tools=[
            "metasploit", "cobalt strike", "gophish",
            "evilginx", "setoolkit", "sqlmap", "routersploit",
            "nuclei", "trevorpray"
        ],
        requires=["Reconnaissance", "Resource Development"],
        leads_to=["Execution", "Persistence", "Privilege Escalation"],
        techniques=["T1190", "T1566", "T1078", "T1133", "T1189", "T1091"]
    ),

    "Execution": MITRETactic(
        name="Execution",
        tactic_id="TA0002",
        keywords=[
            "execute", "run", "launch", "invoke", "call",
            "powershell", "cmd", "command", "script", "payload",
            "execute code", "run command", "invoke command",
            "shellcode", "macro", "vba", "javascript execution",
            "container exec", "scheduled task", "api call", "user execution"
        ],
        tools=[
            "powershell", "cmd", "wmic", "mshta", "regsvr32",
            "rundll32", "cscript", "wscript", "python", "perl",
            "bash", "sh", "lolbas", "certutil"
        ],
        requires=["Initial Access"],
        leads_to=["Persistence", "Privilege Escalation", "Defense Evasion"],
        techniques=["T1059", "T1047", "T1106", "T1203", "T1204", "T1053", "T1569"]
    ),

    "Persistence": MITRETactic(
        name="Persistence",
        tactic_id="TA0003",
        keywords=[
            "persist", "persistence", "maintain", "maintain access",
            "startup", "registry", "scheduled task", "cron", "service",
            "backdoor", "implant", "bootkit", "rootkit",
            "autorun", "logon script", "dll hijack", "browser extension",
            "account manipulation", "ssh key", "golden ticket", "silver ticket"
        ],
        tools=[
            "schtasks", "at", "sc", "reg", "wmi", "autoruns",
            "impacket", "powersploit", "empire", "sharppersist",
            "mimikatz" # Golden Ticket
        ],
        requires=["Execution"],
        leads_to=["Privilege Escalation", "Defense Evasion"],
        techniques=["T1053", "T1547", "T1543", "T1574", "T1078", "T1098", "T1556"]
    ),

    "Privilege Escalation": MITRETactic(
        name="Privilege Escalation",
        tactic_id="TA0004",
        keywords=[
            "escalate", "privesc", "privilege escalation", "elevate",
            "root", "admin", "administrator", "system", "nt authority",
            "sudo", "uac", "uac bypass", "token", "impersonate",
            "exploit elevation", "kernel exploit", "setuid", "capabilities",
            "ad cs", "certificate template"
        ],
        tools=[
            "mimikatz", "powersploit", "metasploit",
            "printspoofer", "godpotato", "sweetpotato",
            "winpeas", "linpeas", "gtfobins", "traitor",
            "certipy", "rubeus" # AD CS abuse
        ],
        requires=["Execution"],
        leads_to=["Credential Access", "Defense Evasion", "Lateral Movement"],
        techniques=["T1068", "T1134", "T1548", "T1078", "T1574", "T1649"]
    ),

    "Defense Evasion": MITRETactic(
        name="Defense Evasion",
        tactic_id="TA0005",
        keywords=[
            "bypass", "evade", "evasion", "hide", "obfuscate",
            "amsi", "amsi bypass", "edr", "av", "antivirus",
            "detection", "disable", "unload", "unhook",
            "process injection", "masquerade", "clear logs",
            "timestomp", "rootkit", "modify registry", "bring your own driver"
        ],
        tools=[
            "invoke-obfuscation", "veil", "shellter", "metasploit",
            "donut", "scarecrow", "chimera", "amsi.fail",
            "mimikatz", "edrsandblast", "backstab"
        ],
        requires=["Execution"],
        leads_to=["Credential Access", "Discovery", "Lateral Movement"],
        techniques=["T1562", "T1140", "T1027", "T1055", "T1070", "T1218", "T1211"]
    ),

    "Credential Access": MITRETactic(
        name="Credential Access",
        tactic_id="TA0006",
        keywords=[
            "credential", "password", "hash", "dump", "extract",
            "lsass", "sam", "ntds", "ntlm", "kerberos",
            "mimikatz", "ticket", "secretsdump", "keylog",
            "steal credential", "harvest password", "crack hash",
            "roast", "as-rep", "brute force", "dcsync", "lsassy"
        ],
        tools=[
            "mimikatz", "pypykatz", "lazagne", "impacket",
            "crackmapexec", "netexec", "rubeus", "sharphound", 
            "procdump", "dumpert", "nanodump", "hashcat", "john",
            "lsassy", "certipy"
        ],
        requires=["Execution", "Privilege Escalation"],
        leads_to=["Lateral Movement", "Collection"],
        techniques=["T1003", "T1558", "T1110", "T1555", "T1552", "T1649"]
    ),

    "Discovery": MITRETactic(
        name="Discovery",
        tactic_id="TA0007",
        keywords=[
            "discover", "enumerate", "recon", "reconnaissance",
            "scan", "map", "identify", "find", "list",
            "whoami", "net user", "net group", "ldap query",
            "network scan", "port scan", "service discovery",
            "system info", "domain trust", "cloud assets", "s3 buckets"
        ],
        tools=[
            "nmap", "masscan", "bloodhound", "sharphound",
            "powerview", "adrecon", "pingcastle", "nessus",
            "seatbelt", "kbt", "winpeas", "linpeas",
            "roadtools", "scoutsuite" # Cloud discovery
        ],
        requires=["Execution"],
        leads_to=["Lateral Movement", "Collection", "Credential Access"],
        techniques=["T1087", "T1018", "T1069", "T1046", "T1083", "T1082", "T1482"]
    ),

    "Lateral Movement": MITRETactic(
        name="Lateral Movement",
        tactic_id="TA0008",
        keywords=[
            "lateral", "pivot", "move", "spread", "propagate",
            "psexec", "wmi", "rdp", "smb", "winrm", "ssh",
            "remote execution", "jump", "move between systems",
            "pass the hash", "pass the ticket", "overpass the hash",
            "dcom", "remote service", "tunneling", "proxy"
        ],
        tools=[
            "psexec", "smbexec", "wmiexec", "netexec",
            "impacket", "covenant", "cobalt strike", "metasploit",
            "ligolo-ng", "chisel", "ssh", "evil-winrm"
        ],
        requires=["Credential Access", "Execution"],
        leads_to=["Collection", "Exfiltration"],
        techniques=["T1021", "T1570", "T1550", "T1563", "T1091", "T1572"]
    ),

    "Collection": MITRETactic(
        name="Collection",
        tactic_id="TA0009",
        keywords=[
            "collect", "gather", "harvest", "screenshot",
            "keylog", "clipboard", "record", "capture",
            "archive", "compress", "stage data", "file collection",
            "email collection", "database dump", "browser stealing"
        ],
        tools=[
            "7zip", "winrar", "rclone", "sharpshares",
            "snaffler", "powersploit", "empire", "keylogger",
            "lazagne"
        ],
        requires=["Execution"],
        leads_to=["Exfiltration"],
        techniques=["T1005", "T1039", "T1113", "T1115", "T1560", "T1119", "T1114"]
    ),

    "Command and Control": MITRETactic(
        name="Command and Control",
        tactic_id="TA0011",
        keywords=[
            "c2", "c&c", "command and control", "beacon", "callback",
            "communication", "channel", "reverse shell", "bind shell",
            "http c2", "dns c2", "covert channel", "domain fronting",
            "doh", "web shell", "teamserver", "agent"
        ],
        tools=[
            "cobalt strike", "covenant", "empire", "metasploit",
            "sliver", "mythic", "poshc2", "merlin", "havoc",
            "brute ratel", "shad0w", "villain"
        ],
        requires=["Execution"],
        leads_to=["Exfiltration", "Lateral Movement", "Impact"],
        techniques=["T1071", "T1573", "T1090", "T1095", "T1105", "T1102"]
    ),

    "Exfiltration": MITRETactic(
        name="Exfiltration",
        tactic_id="TA0010",
        keywords=[
            "exfil", "exfiltrate", "steal", "extract", "transfer",
            "upload", "send", "egress", "data theft",
            "c2 channel", "dns tunnel", "http upload", "cloud transfer"
        ],
        tools=[
            "rclone", "mega", "dropbox", "anonfiles",
            "dnscat2", "iodine", "cobalt strike", "curl", "wget",
            "chisel"
        ],
        requires=["Collection", "Command and Control"],
        leads_to=[],
        techniques=["T1041", "T1048", "T1567", "T1020", "T1052"]
    ),

    "Impact": MITRETactic(
        name="Impact",
        tactic_id="TA0040",
        keywords=[
            "destroy", "delete", "wipe", "corrupt", "encrypt",
            "ransom", "ransomware", "disable", "dos", "denial",
            "manipulate", "defacement", "inhibit recovery",
            "shadow copy delete", "account lockout"
        ],
        tools=[
            "ryuk", "lockbit", "conti", "blackcat",
            "ransomware", "wannacry", "notpetya", "vssadmin",
            "cipher", "wiper"
        ],
        requires=["Execution", "Privilege Escalation"],
        leads_to=[],
        techniques=["T1486", "T1485", "T1490", "T1491", "T1561", "T1489"]
    )
}

ATTACK_CHAINS = {
    "credential_dumping_chain": [
        "Initial Access",
        "Execution",
        "Privilege Escalation",
        "Credential Access"
    ],
    "lateral_movement_chain": [
        "Credential Access",
        "Discovery",
        "Lateral Movement",
        "Execution"
    ],
    "ransomware_chain": [
        "Initial Access",
        "Execution",
        "Privilege Escalation",
        "Defense Evasion",
        "Impact"
    ],
    "data_exfiltration_chain": [
        "Initial Access",
        "Execution",
        "Discovery",
        "Collection",
        "Exfiltration"
    ],
    "ad_cs_compromise": [
        "Initial Access", 
        "Discovery",  # Finding the PKI
        "Privilege Escalation", # ESC1/ESC8 exploitation
        "Credential Access" # Golden Cert
    ],
    "cloud_breach_chain": [
        "Reconnaissance",
        "Initial Access",
        "Discovery", # Cloud enumeration
        "Exfiltration" # S3 bucket dump
    ],
    "full_kill_chain": [
        "Reconnaissance",
        "Resource Development",
        "Initial Access",
        "Command and Control",
        "Execution",
        "Persistence",
        "Lateral Movement",
        "Exfiltration"
    ]
}


TOOL_TO_TECHNIQUES = {
    # Credential Access Tools
    "mimikatz": ["T1003.001", "T1003.002", "T1558.003", "T1207", "T1098", "T1555"],
    "pypykatz": ["T1003.001"],
    "lazagne": ["T1555.003", "T1552"],
    "secretsdump": ["T1003.002", "T1003.003", "T1003.004"],
    "rubeus": ["T1558", "T1558.003", "T1558.004", "T1550.003"],
    "certipy": ["T1649", "T1558", "T1098"], # Critical AD CS tool
    "procdump": ["T1003.001"],
    "lsassy": ["T1003.001"],
    "hashcat": ["T1110.002"],

    # Lateral Movement / Remote Exec
    "psexec": ["T1021.002", "T1570", "T1569.002"],
    "wmiexec": ["T1021.006", "T1047"],
    "smbexec": ["T1021.002"],
    "crackmapexec": ["T1021.002", "T1021.006", "T1003", "T1110.003"],
    "netexec": ["T1021.002", "T1021.006", "T1003", "T1110.003"], # Successor to CME
    "impacket": ["T1021", "T1003", "T1558", "T1098", "T1047"],
    "evil-winrm": ["T1021.006", "T1059.001"],
    "ssh": ["T1021.004"],
    "ligolo-ng": ["T1090", "T1572", "T1021"], # Pivoting/Tunneling
    "chisel": ["T1090", "T1572", "T1021"],

    # C2 Frameworks
    "cobalt strike": ["T1071", "T1090", "T1055", "T1105", "T1562.001"],
    "empire": ["T1059.001", "T1071", "T1053", "T1555"],
    "covenant": ["T1071", "T1059"],
    "metasploit": ["T1059", "T1190", "T1203", "T1210", "T1021"],
    "sliver": ["T1071", "T1059", "T1055"],
    "havoc": ["T1071", "T1059", "T1055"],
    "mythic": ["T1071", "T1059"],

    # Defense Evasion / Persistence
    "amsi.fail": ["T1562.001"],
    "invoke-obfuscation": ["T1027", "T1059.001"],
    "donut": ["T1055"],
    "veil": ["T1027"],
    "schtasks": ["T1053.005"],
    "reg": ["T1547.001", "T1574", "T1112"],
    "edrsandblast": ["T1562.001", "T1003.001"],

    # Discovery
    "bloodhound": ["T1087.002", "T1069.002", "T1083", "T1482"],
    "sharphound": ["T1087.002", "T1069.002", "T1083", "T1482"],
    "powerview": ["T1087.002", "T1018", "T1069", "T1482"],
    "adrecon": ["T1087.002", "T1069.002"],
    "seatbelt": ["T1082", "T1012", "T1562.001"],
    "linpeas": ["T1082", "T1068", "T1003", "T1548"],
    "winpeas": ["T1082", "T1068", "T1003", "T1548"],
    "snaffler": ["T1083", "T1552.001"], # Finds creds in files
    "roadtools": ["T1087.004", "T1069.003"], # Azure discovery

    # Reconnaissance / Ingress
    "nmap": ["T1046", "T1595.002"],
    "masscan": ["T1046", "T1595.002"],
    "shodan": ["T1592.002"],
    "nuclei": ["T1595.002", "T1190"],
    "amass": ["T1590.002"],
    "certutil": ["T1105"],
    "curl": ["T1105"],
    "wget": ["T1105"]
}

def get_tactics_for_query(query_lower: str) -> Set[str]:
    """
    Detect MITRE tactics from natural language query.

    Args:
        query_lower: Lowercased query text

    Returns:
        Set of detected tactic names
    """
    detected_tactics = set()

    for tactic_name, tactic in MITRE_TACTICS.items():
        # Check if any keyword matches
        for keyword in tactic.keywords:
            if keyword in query_lower:
                detected_tactics.add(tactic_name)
                break

    return detected_tactics


def get_tools_for_query(query_lower: str) -> Set[str]:
    """
    Detect cybersecurity tools mentioned in query.

    Args:
        query_lower: Lowercased query text

    Returns:
        Set of detected tool names
    """
    detected_tools = set()

    for tactic in MITRE_TACTICS.values():
        for tool in tactic.tools:
            if tool.lower() in query_lower:
                detected_tools.add(tool)

    return detected_tools


def expand_with_attack_chain(primary_tactics: Set[str]) -> Dict[str, float]:
    """
    Expand detected tactics using attack chain relationships.

    Returns dict of {tactic_name: boost_weight}
    - Primary tactics: 1.0 (full boost)
    - Required tactics: 0.4 (needs this to execute primary)
    - Leads-to tactics: 0.3 (common next step)

    Args:
        primary_tactics: Tactics directly detected from query

    Returns:
        Dict mapping tactic names to boost weights
    """
    tactic_weights = {}

    # Primary tactics get full weight
    for tactic in primary_tactics:
        tactic_weights[tactic] = 1.0

    # Add prerequisites and next steps with partial boost
    for tactic_name in primary_tactics:
        if tactic_name not in MITRE_TACTICS:
            continue

        tactic_obj = MITRE_TACTICS[tactic_name]

        # Prerequisites (e.g., need Execution for Lateral Movement)
        for req_tactic in tactic_obj.requires:
            if req_tactic not in tactic_weights:
                tactic_weights[req_tactic] = 0.4

        # Common next steps
        for next_tactic in tactic_obj.leads_to:
            if next_tactic not in tactic_weights:
                tactic_weights[next_tactic] = 0.3

    return tactic_weights


def get_techniques_for_tools(tools: Set[str]) -> Set[str]:
    """
    Map detected tools to MITRE technique IDs.

    Args:
        tools: Set of tool names

    Returns:
        Set of technique IDs (e.g., T1003, T1558)
    """
    techniques = set()

    for tool in tools:
        tool_lower = tool.lower()
        if tool_lower in TOOL_TO_TECHNIQUES:
            techniques.update(TOOL_TO_TECHNIQUES[tool_lower])

    return techniques
