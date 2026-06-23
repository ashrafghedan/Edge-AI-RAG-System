from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .utils import normalize_whitespace


_CATEGORY_PRIORITY = {
    'self_harm': 0,
    'sensitive_data': 1,
    'prompt_injection': 2,
    'security_bypass': 3,
    'malicious_action': 4,
}

_CATEGORY_MESSAGES = {
    'self_harm': (
        "I'm really sorry you're dealing with this. I can't help with instructions that could be used for self-harm. "
        'If you might act on these thoughts, call emergency services now. '
        "If you're in the U.S. or its territories, call or text 988 right now to reach the 988 Suicide & Crisis Lifeline. "
        "If you're elsewhere, contact your local emergency number or crisis line now. "
        'If you can, move the rope or other means away from you and get near another person while you reach out for help.'
    ),
    'sensitive_data': 'I cannot help reveal or extract real secrets, credentials, or private account data.',
    'prompt_injection': 'I cannot expose hidden system prompts, internal instructions, or bypass the instruction hierarchy.',
    'security_bypass': 'I cannot help bypass authentication or break through real security controls.',
    'malicious_action': 'I cannot help with instructions for unauthorized access, phishing, malware, or data theft.',
}

_PURPOSE_CATEGORIES = {
    'user_query': {'self_harm', 'sensitive_data', 'prompt_injection', 'security_bypass', 'malicious_action'},
    'user_answer': {'self_harm', 'sensitive_data', 'prompt_injection', 'security_bypass', 'malicious_action'},
    'generated_content': {'self_harm', 'sensitive_data', 'prompt_injection', 'security_bypass', 'malicious_action'},
    'model_output': {'self_harm', 'sensitive_data', 'prompt_injection', 'security_bypass', 'malicious_action'},
    'retrieved_context': {'sensitive_data', 'prompt_injection'},
}

_ACCESS_ACTIONS = {
    'cat',
    'display',
    'dump',
    'expose',
    'extract',
    'give me',
    'leak',
    'list',
    'open',
    'print',
    'read',
    'repeat',
    'reveal',
    'send me',
    'show me',
    'summarize',
    'tell me',
}
_DIRECT_REQUEST_HINTS = {
    'can you',
    'get me',
    'help me',
    'how do i',
    'how to',
    'i need',
    'show me',
    'walk me through',
}
_EDUCATIONAL_PREFIXES = (
    'analyze ',
    'compare ',
    'define ',
    'describe ',
    'discuss ',
    'explain ',
    'outline ',
    'summarize ',
    'teach me ',
    'what are ',
    'what is ',
    'why is ',
)
_EDUCATIONAL_HINTS = {
    'concept',
    'definition',
    'educational',
    'example',
    'examples',
    'for a lesson',
    'for class',
    'overview',
    'the difference between',
    'what does it mean',
    'why is',
}
_CREATIVE_VERBS = {'compose', 'create', 'draft', 'generate', 'tell', 'write'}
_CREATIVE_NOUNS = {'dialogue', 'fiction', 'fictional', 'narrative', 'novel', 'poem', 'scenario', 'scene', 'story'}
_SECRET_TARGETS = {
    '.env',
    'access token',
    'api key',
    'auth token',
    'bank account',
    'bearer token',
    'connection string',
    'credential',
    'credentials',
    'credit card',
    'cvv',
    'database uri',
    'database url',
    'debit card',
    'environment variables',
    'env vars',
    'iban',
    'jwt',
    'national id',
    'passport number',
    'passcode',
    'password',
    'pin',
    'private key',
    'recovery phrase',
    'refresh token',
    'routing number',
    'seed phrase',
    'session cookie',
    'social security',
    'ssh key',
    'ssn',
    'swift',
}
_SYSTEM_SECRET_TARGETS = {
    '.env',
    'access token',
    'api key',
    'auth token',
    'bearer token',
    'connection string',
    'credential',
    'credentials',
    'database uri',
    'database url',
    'environment variables',
    'env vars',
    'jwt',
    'private key',
    'recovery phrase',
    'refresh token',
    'seed phrase',
    'session cookie',
    'ssh key',
}
_INTERNAL_TARGETS = {
    'developer message',
    'developer prompt',
    'developer notes',
    'hidden instructions',
    'hidden prompt',
    'internal instructions',
    'internal prompt',
    'internal rules',
    'system instructions',
    'system prompt',
}
_GUARDRAIL_TARGETS = {'filters', 'guardrails', 'policy', 'safety'}
_SECURITY_BYPASS_TERMS = {
    '2fa',
    'authentication',
    'bypass auth',
    'bypass authentication',
    'bypass login',
    'captcha',
    'credential stuffing',
    'disable mfa',
    'disable 2fa',
    'escalate privileges',
    'exploit',
    'exploit chain',
    'exploit kit',
    'login page',
    'mfa',
    'privilege escalation',
    'session hijack',
    'session fixation',
    'sql injection',
    'sqli',
    'token theft',
    'xss',
}
_INSTRUCTIONAL_HINTS = {
    'attack chain',
    'build a',
    'code for',
    'command to',
    'commands for',
    'give me code',
    'give me steps',
    'how do i',
    'how to',
    'payload',
    'script',
    'step by step',
    'steps to',
    'walk me through',
    'write code',
}
_MALICIOUS_ACTION_TERMS = {
    'botnet',
    'credential theft',
    'data exfiltration',
    'exfiltrate data',
    'keylogger',
    'malware',
    'phishing',
    'phishing email',
    'phishing kit',
    'ransomware',
    'rat',
    'remote access trojan',
    'steal credentials',
    'steal passwords',
    'steal data',
    'trojan',
}
_MALICIOUS_BUILD_HINTS = {'create', 'draft', 'generate', 'template', 'write'}
_REAL_WORLD_TARGET_HINTS = {
    '.env',
    'account',
    'admin panel',
    'company',
    'customer',
    'database',
    'production',
    'router',
    'server',
    'system',
    'vpn',
    'website',
    'wifi',
}
_EXFILTRATION_HINTS = {'download', 'export', 'steal', 'copy', 'exfiltrate'}
_OVERRIDE_PATTERNS = [
    r'(?:^|[.!?]\s*)(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b',
    r'\b(?:jailbreak|do anything now|uncensored mode)\b',
    r'\bpretend you have no restrictions\b',
    r'\bno restrictions\b',
    r'\bno rules\b',
    r'\boverride\b.*\b(?:safety|guardrails|policy|filters)\b',
]
_SENSITIVE_LITERAL_PATTERNS = [
    r'\bsk-[A-Za-z0-9]{20,}\b',
    r'\bghp_[A-Za-z0-9]{30,}\b',
    r'\bAKIA[0-9A-Z]{16}\b',
    r'\b(?:\d{4}[- ]?){3}\d{4}\b',
    r'\b\d{3}-\d{2}-\d{4}\b',
]
_SELF_HARM_EXPLICIT_PATTERNS = [
    r'\b(?:kill myself|end my life|take my life|commit suicide|suicide plan|want to die|don\'t want to live)\b',
    r'\b(?:hang myself|strangle myself|overdose|cut myself|hurt myself|harm myself)\b',
]
_SELF_HARM_HANGING_METHOD_PATTERNS = [
    r'\b(?:how\s+(?:do\s+i|to)|help me|show me|instructions?|steps?|step\s+by\s+step)\b.{0,90}\b(?:tie|knot|loop|noose|rope)\b.{0,90}\b(?:neck|throat|hang)\b',
    r'\b(?:tie|knot|loop|noose|rope)\b.{0,90}\b(?:around|fit|fits|tight|perfectly)\b.{0,90}\b(?:neck|throat)\b',
    r'\b(?:rope|noose|knot|loop)\b.{0,90}\b(?:so\s+i\s+(?:can|could|will|would)|to)\s+hang\b',
]
_SELF_HARM_DISTRESS_HINTS = {
    'alone',
    "can't go on",
    'cant go on',
    'hopeless',
    'i am done',
    'i feel empty',
    'i give up',
    'lonely',
    'no reason to live',
    'not safe',
    'worthless',
}
_SELF_HARM_MEANS_HINTS = {
    'blade',
    'bridge',
    'gun',
    'hang it',
    'hang myself',
    'hanging',
    'knife',
    'noose',
    'pills',
    'place to hang',
    'poison',
    'rope',
}
_SELF_HARM_ACTION_HINTS = {
    'help me',
    'how do i',
    'how to',
    'instructions',
    'show me',
    'step by step',
    'tie a rope',
    'tie the rope',
    'what knot',
}
_SELF_HARM_BODY_HINTS = {
    'around my neck',
    'around the neck',
    'around your neck',
    'my neck',
    'neck',
    'throat',
}
_SELF_HARM_OUTPUT_HINTS = {
    'around your neck',
    'hang it',
    'hang yourself',
    'knot',
    'loop',
    'noose',
    'rope',
    'step 1',
    'step-by-step',
    'tie a rope',
}
_SELF_HARM_OUTPUT_DANGER_HINTS = {
    'around your neck',
    'hang it',
    'hang yourself',
    'noose',
}
_SELF_HARM_OUTPUT_METHOD_PATTERNS = [
    r'\b(?:tie|knot|loop|noose|rope)\b.{0,90}\b(?:neck|throat|hang|pull\s+tight)\b',
]
_INSTRUCTIONAL_LIST_PATTERNS = [
    r'(?m)^\s*(?:\d+\.|[-*])\s+',
    r'\b(?:step\s+1|step\s+2|first,|next,|finally,)\b',
    r'\b(?:here is|here\'s)\s+(?:a\s+)?(?:simple\s+)?method\b',
]


@dataclass(slots=True)
class SecurityFinding:
    category: str
    reason: str
    confidence: float = 1.0


@dataclass(slots=True)
class SecurityDecision:
    blocked: bool
    categories: list[str] = field(default_factory=list)
    findings: list[SecurityFinding] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def primary_category(self) -> str | None:
        return self.categories[0] if self.categories else None


class LocalSecurityPolicy:
    def evaluate_text(self, text: str, *, purpose: str) -> SecurityDecision:
        normalized = normalize_whitespace(text).lower()
        if not normalized:
            return SecurityDecision(blocked=False)

        allowed_categories = _PURPOSE_CATEGORIES.get(purpose, _PURPOSE_CATEGORIES['user_query'])
        findings: list[SecurityFinding] = []

        evaluators = (
            self._evaluate_self_harm,
            self._evaluate_sensitive_data,
            self._evaluate_prompt_injection,
            self._evaluate_security_bypass,
            self._evaluate_malicious_action,
        )
        for evaluator in evaluators:
            finding = evaluator(text, normalized, purpose=purpose)
            if finding is None:
                continue
            if finding.category not in allowed_categories:
                continue
            if finding.confidence < 0.75:
                continue
            findings.append(finding)

        findings.sort(key=lambda item: (_CATEGORY_PRIORITY.get(item.category, 99), -item.confidence))
        categories = list(dict.fromkeys(finding.category for finding in findings))
        confidence = max((finding.confidence for finding in findings), default=0.0)
        return SecurityDecision(
            blocked=bool(categories),
            categories=categories,
            findings=findings,
            confidence=confidence,
        )

    def refusal_message(self, decision: SecurityDecision) -> str:
        if decision.primary_category is None:
            return 'This request is blocked.'
        base = _CATEGORY_MESSAGES.get(decision.primary_category, 'This request is blocked.')
        if decision.primary_category == 'self_harm':
            return base
        if not decision.findings:
            return base
        detail = decision.findings[0].reason.rstrip('.')
        return f'{base} {detail}.'

    def is_refusal_text(self, text: str) -> bool:
        normalized = normalize_whitespace(text)
        if not normalized:
            return False
        for message in _CATEGORY_MESSAGES.values():
            base = normalize_whitespace(message)
            if normalized == base or normalized.startswith(base + ' '):
                return True
        return False

    def audit_preview(self, text: str, *, max_length: int = 160) -> str:
        preview = normalize_whitespace(text)
        if len(preview) <= max_length:
            return preview
        return preview[: max_length - 3].rstrip() + '...'

    def _evaluate_sensitive_data(
        self,
        text: str,
        normalized: str,
        *,
        purpose: str,
    ) -> SecurityFinding | None:
        literal_match = self._first_pattern_match(text, _SENSITIVE_LITERAL_PATTERNS)
        if literal_match is not None:
            preview = literal_match.group(0)
            if len(preview) > 12:
                preview = preview[:8] + '...'
            return SecurityFinding(
                category='sensitive_data',
                reason=f'The content includes a real secret or private identifier ({preview}).',
                confidence=0.99,
            )

        target_hits = self._find_phrase_hits(normalized, _SECRET_TARGETS)
        access_hits = self._find_phrase_hits(normalized, _ACCESS_ACTIONS)
        exfiltration_hits = self._find_phrase_hits(normalized, _EXFILTRATION_HINTS)
        real_world_hits = self._find_phrase_hits(normalized, _REAL_WORLD_TARGET_HINTS)
        if not target_hits or not (access_hits or exfiltration_hits):
            return None

        if self._is_educational_request(normalized) and not self._looks_like_direct_request(normalized):
            return None
        if self._is_creative_request(normalized):
            instructional_hits = self._find_phrase_hits(normalized, _INSTRUCTIONAL_HINTS)
            if not instructional_hits and not real_world_hits:
                return None
        if purpose in {'generated_content', 'model_output'} and not literal_match:
            system_secret_hits = [item for item in target_hits if item in _SYSTEM_SECRET_TARGETS]
            if not system_secret_hits and not real_world_hits:
                return None

        target_label = self._format_hits(target_hits)
        access_label = self._format_hits(access_hits or exfiltration_hits)
        confidence = 0.9 if (real_world_hits or '.env' in target_hits or 'environment variables' in target_hits) else 0.82
        return SecurityFinding(
            category='sensitive_data',
            reason=f'The request asks to {access_label} real sensitive data such as {target_label}',
            confidence=confidence,
        )

    def _evaluate_self_harm(
        self,
        text: str,
        normalized: str,
        *,
        purpose: str,
    ) -> SecurityFinding | None:
        if self._is_educational_request(normalized) and not self._looks_like_direct_request(normalized):
            return None

        explicit_self_harm = self._matches_any(normalized, _SELF_HARM_EXPLICIT_PATTERNS)
        if explicit_self_harm and purpose != 'model_output':
            return SecurityFinding(
                category='self_harm',
                reason='The request indicates possible self-harm intent or asks for self-harm methods',
                confidence=0.99,
            )

        distress_hits = self._find_phrase_hits(normalized, _SELF_HARM_DISTRESS_HINTS)
        means_hits = self._find_phrase_hits(normalized, _SELF_HARM_MEANS_HINTS)
        action_hits = self._find_phrase_hits(normalized, _SELF_HARM_ACTION_HINTS)
        body_hits = self._find_phrase_hits(normalized, _SELF_HARM_BODY_HINTS)
        method_request = self._matches_any(normalized, _SELF_HARM_HANGING_METHOD_PATTERNS) or (
            means_hits and action_hits and body_hits
        )
        if method_request and purpose != 'model_output':
            return SecurityFinding(
                category='self_harm',
                reason='The request asks for actionable self-harm method instructions',
                confidence=0.98,
            )
        if self._is_creative_request(normalized) and purpose != 'model_output':
            return None
        if distress_hits and means_hits and action_hits:
            return SecurityFinding(
                category='self_harm',
                reason='The request combines distress, available means, and a request for harmful instructions',
                confidence=0.97,
            )

        if purpose == 'model_output':
            output_hits = self._find_phrase_hits(normalized, _SELF_HARM_OUTPUT_HINTS)
            danger_hits = self._find_phrase_hits(normalized, _SELF_HARM_OUTPUT_DANGER_HINTS)
            has_instructional_shape = self._matches_any(text, _INSTRUCTIONAL_LIST_PATTERNS)
            has_method_wording = self._matches_any(normalized, _SELF_HARM_OUTPUT_METHOD_PATTERNS)
            if has_instructional_shape and (has_method_wording or (output_hits and danger_hits)):
                return SecurityFinding(
                    category='self_harm',
                    reason='The generated response contains actionable self-harm instructions',
                    confidence=0.94,
                )
        return None

    def _evaluate_prompt_injection(
        self,
        _text: str,
        normalized: str,
        *,
        purpose: str,
    ) -> SecurityFinding | None:
        internal_hits = self._find_phrase_hits(normalized, _INTERNAL_TARGETS)
        access_hits = self._find_phrase_hits(normalized, _ACCESS_ACTIONS)
        guardrail_hits = self._find_phrase_hits(normalized, _GUARDRAIL_TARGETS)
        override_hit = self._matches_any(normalized, _OVERRIDE_PATTERNS)
        if not override_hit and not (internal_hits and access_hits):
            return None
        if purpose in {'generated_content', 'model_output'} and override_hit and not internal_hits and not guardrail_hits:
            return None

        if self._is_educational_request(normalized) and not self._looks_like_direct_request(normalized):
            return None
        if self._is_creative_request(normalized):
            instructional_hits = self._find_phrase_hits(normalized, _INSTRUCTIONAL_HINTS)
            if not instructional_hits:
                return None

        if internal_hits and access_hits:
            target_label = self._format_hits(internal_hits)
            return SecurityFinding(
                category='prompt_injection',
                reason=f'The request tries to expose hidden system data such as {target_label}',
                confidence=0.92,
            )

        label = self._format_hits(guardrail_hits) or 'the assistant safeguards'
        return SecurityFinding(
            category='prompt_injection',
            reason=f'The request attempts to override or disable {label}',
            confidence=0.84,
        )

    def _evaluate_security_bypass(
        self,
        _text: str,
        normalized: str,
        *,
        purpose: str,
    ) -> SecurityFinding | None:
        bypass_hits = self._find_phrase_hits(normalized, _SECURITY_BYPASS_TERMS)
        instructional_hits = self._find_phrase_hits(normalized, _INSTRUCTIONAL_HINTS)
        real_world_hits = self._find_phrase_hits(normalized, _REAL_WORLD_TARGET_HINTS)
        if not bypass_hits:
            return None

        if self._is_educational_request(normalized) and not (instructional_hits or self._looks_like_direct_request(normalized)):
            return None
        if self._is_creative_request(normalized) and not instructional_hits:
            return None
        if not (instructional_hits or real_world_hits or self._looks_like_direct_request(normalized)):
            return None

        bypass_label = self._format_hits(bypass_hits)
        confidence = 0.9 if (real_world_hits or 'bypass auth' in bypass_hits or 'bypass authentication' in bypass_hits) else 0.8
        return SecurityFinding(
            category='security_bypass',
            reason=f'The request asks for help bypassing or exploiting {bypass_label}',
            confidence=confidence,
        )

    def _evaluate_malicious_action(
        self,
        _text: str,
        normalized: str,
        *,
        purpose: str,
    ) -> SecurityFinding | None:
        malicious_hits = self._find_phrase_hits(normalized, _MALICIOUS_ACTION_TERMS)
        if not malicious_hits:
            return None

        build_hits = self._find_phrase_hits(normalized, _MALICIOUS_BUILD_HINTS)
        instructional_hits = self._find_phrase_hits(normalized, _INSTRUCTIONAL_HINTS)
        if self._is_educational_request(normalized) and not (build_hits or instructional_hits):
            return None
        if self._is_creative_request(normalized) and not instructional_hits and not self._looks_like_artifact_request(normalized):
            return None
        if not (build_hits or instructional_hits or self._looks_like_direct_request(normalized)):
            return None

        label = self._format_hits(malicious_hits)
        confidence = 0.9 if self._looks_like_artifact_request(normalized) else 0.82
        return SecurityFinding(
            category='malicious_action',
            reason=f'The request asks for actionable assistance with {label}',
            confidence=confidence,
        )

    @staticmethod
    def _first_pattern_match(text: str, patterns: list[str]) -> re.Match[str] | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match is not None:
                return match
        return None

    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        if phrase.startswith('.'):
            return phrase in text
        return re.search(rf'\b{re.escape(phrase)}\b', text, flags=re.IGNORECASE) is not None

    def _find_phrase_hits(self, text: str, phrases: set[str]) -> list[str]:
        return sorted(phrase for phrase in phrases if self._contains_phrase(text, phrase))

    @staticmethod
    def _format_hits(hits: list[str]) -> str:
        if not hits:
            return ''
        if len(hits) == 1:
            return hits[0]
        return ', '.join(hits[:2])

    @staticmethod
    def _is_educational_request(normalized: str) -> bool:
        if normalized.startswith(_EDUCATIONAL_PREFIXES):
            return True
        return any(hint in normalized for hint in _EDUCATIONAL_HINTS)

    @staticmethod
    def _is_creative_request(normalized: str) -> bool:
        has_verb = any(re.search(rf'\b{re.escape(verb)}\b', normalized) for verb in _CREATIVE_VERBS)
        has_noun = any(re.search(rf'\b{re.escape(noun)}\b', normalized) for noun in _CREATIVE_NOUNS)
        return has_verb and has_noun

    @staticmethod
    def _looks_like_direct_request(normalized: str) -> bool:
        if any(normalized.startswith(prefix) for prefix in _EDUCATIONAL_PREFIXES):
            return False
        return any(hint in normalized for hint in _DIRECT_REQUEST_HINTS)

    @staticmethod
    def _looks_like_artifact_request(normalized: str) -> bool:
        return any(
            phrase in normalized
            for phrase in (
                'phishing email',
                'phishing kit',
                'keylogger',
                'payload',
                'script',
                'template',
            )
        )


@lru_cache(maxsize=1)
def get_default_security_policy() -> LocalSecurityPolicy:
    return LocalSecurityPolicy()


def should_refuse_query(question: str) -> bool:
    return get_default_security_policy().evaluate_text(question, purpose='user_query').blocked
