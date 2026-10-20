"""D1 entry, resource and semantic guards. No scientific defaults from .env."""
import contextvars
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'artifacts/paperresearch-v2'
OUT = BASE / 'phase4b-v3-b2-d1-dev-retrieval-implementation'
B2 = BASE / 'phase4b-v3-b2-retrieval-preregistration'
B1 = BASE / 'phase4b-v3-b1-formal-benchmark-construction'
A2 = BASE / 'phase4b-v3-a2-pdf-native-extraction-capacity-check'
PROTOCOL = '9e80c5ec36dd93194770d1fb1db17e580b731059cfe2b7b449975d12a3246549'
CORPUS = '6d53c3873a2c1e71cd8746ba5e687926e99a972bb9ed255d318b945dde9477d8'
IDENTITY = 'PHASE4B_V3_B2_D1_DEV_RETRIEVAL_IMPLEMENTATION'
OPAQUE = contextvars.ContextVar('hash_only', default=False)
DEV_ALLOWED = False
NATIVE_ALLOWED = False
DENIED = []
NETWORK_ALLOWED = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def digest(value):
    return hashlib.sha256(canon(value)).hexdigest()


def sha(path):
    token = OPAQUE.set(True)
    try:
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    finally:
        OPAQUE.reset(token)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def put(relative, value):
    path = (OUT / relative).resolve()
    assert path.is_relative_to(OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.write.tmp')
    temporary.write_bytes(canon(value) + b'\n')
    temporary.replace(path)


def audit(event, args):
    if event == 'socket.connect' and not NETWORK_ALLOWED:
        raise PermissionError('D1 network inference forbidden; all model files local')
    if event != 'open' or isinstance(args[0], int):
        return
    path, mode, flags = args
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(path)).resolve()
    writing = any(v in (mode or '') for v in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    if writing and not path.is_relative_to(OUT):
        raise PermissionError('D1_WRITE_BOUNDARY_VIOLATION')
    if writing or OPAQUE.get():
        return
    if path.is_relative_to(B1):
        allowed = {'decision.json', 'entry_audit.json', 'completion_freeze.json', 'accepted/manifest.json', 'accepted/manifest_digest.json',
                   'split/dev_manifest.json', 'split/test_manifest.json', 'split/test_seal.json', 'sealed/construction_seal.json'}
        if DEV_ALLOWED:
            allowed.add('split/dev.jsonl')
        if path.relative_to(B1).as_posix() not in allowed:
            DENIED.append(path.relative_to(ROOT).as_posix())
            raise PermissionError('D1_SEALED_SEMANTIC_ACCESS_DENIED')
    if path.is_relative_to(A2 / 'extraction/native') and not NATIVE_ALLOWED:
        DENIED.append(path.relative_to(ROOT).as_posix())
        raise PermissionError('D1_NATIVE_INDEX_INPUT_BEFORE_MODEL_PREFLIGHT_DENIED')


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, relative in {'TEMP': 'tmp', 'TMP': 'tmp', 'HF_HOME': 'cache/hf', 'HUGGINGFACE_HUB_CACHE': 'cache/hf/hub',
                           'TRANSFORMERS_CACHE': 'cache/hf/transformers', 'TORCH_HOME': 'cache/torch', 'XDG_CACHE_HOME': 'cache',
                           'PIP_CACHE_DIR': 'cache/pip', 'MPLCONFIGDIR': 'cache/matplotlib'}.items():
        target = OUT / relative
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    os.environ.update({'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'DO_NOT_TRACK': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'})
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(OUT / 'dependencies'))
    sys.path.insert(0, str(ROOT / 'src'))
    sys.addaudithook(audit)


def inherited_contract():
    name = 'frozen_B2_contract'
    spec = importlib.util.spec_from_file_location(name, B2 / 'engineering/contract.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resource_accepted(value):
    if value['status'] == 'PASS':
        return True
    if value['status'] != 'PASS_WITH_AUTHORIZED_HISTORICAL_MEASUREMENT_EXCEPTION':
        return False
    exception = value.get('authorized_measurement_exception') or {}
    registration = read(OUT / 'indexes/repair_preregistration.json')
    return (exception.get('identity') == registration['identity'] == 'D1_INDEX_OPERATIONAL_REPAIR_01'
            and registration.get('user_authorization') == '允许该限定计量例外并继续'
            and exception.get('preregistration_SHA256') == sha(OUT / 'indexes/repair_preregistration.json')
            and value['violation'] is None)


def expected_history():
    expected = dict(read(B2 / 'guards/historical_hashes.json')['all_historical_expected_hashes'])
    freeze = read(B2 / 'completion_freeze.json')
    prefix = B2.relative_to(ROOT).as_posix() + '/'
    expected.update({prefix + k: v for k, v in freeze['artifact_sha256_excludes_self'].items()})
    expected[prefix + 'completion_freeze.json'] = sha(B2 / 'completion_freeze.json')
    return expected


def verify_history(expected):
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda p: sha(ROOT / p), expected))
    assert all(expected[p] == h for p, h in zip(expected, values, strict=True)), 'HISTORY_SHA256_MISMATCH'
    return {'status': 'PASS', 'files': len(expected), 'changed': 0, 'expected_map_digest': digest(expected), 'method': 'streaming opaque SHA256; no semantic decode'}


def entry():
    assert not (OUT / 'completion_freeze.json').exists(), 'D1_SEALED_NO_RERUN'
    expected = expected_history()
    print('ENTRY: verify B2/B1/corpus frozen hashes and immutable history', flush=True)
    verified = verify_history(expected)
    freeze = read(B2 / 'completion_freeze.json')
    assert freeze['phase_status'] == 'PHASE4B_V3_B2_RETRIEVAL_PROTOCOL_READY'
    assert freeze['completion_boundary'] == 'READY_FOR_PHASE4B_V3_B2_D1_DEV_RETRIEVAL_IMPLEMENTATION'
    assert freeze['protocol_digest'] == PROTOCOL
    binding = read(B2 / 'protocol/retrieval_protocol_digest.json')
    assert digest(binding['binding']) == binding['digest'] == PROTOCOL
    b1 = read(B1 / 'decision.json')
    assert b1['category'] == 'REDUCED' and b1['split_counts'] == {'dev': 20, 'test': 55}
    assert read(B1 / 'accepted/manifest.json')['case_count'] == 75
    b1hashes = read(B2 / 'guards/b1_manifest_and_seal.json')['exact_file_SHA256']
    assert all(sha(B1 / p) == h for p, h in b1hashes.items())
    assert read(A2 / 'extraction/document_manifest.json')['corpus40_digest'] == CORPUS
    denied = []
    for rel in ['split/test.jsonl', 'sealed/rejected_manifest.json', 'accepted/cases.jsonl', 'construction/attempts.jsonl', 'requests/example/response.json']:
        try:
            with (B1 / rel).open('rb'):
                pass
        except PermissionError:
            denied.append(rel)
    assert len(denied) == 5
    put('guards/test_access_denial.json', {'status': 'PASS', 'before_open_denials': denied, 'TEST_query_reads': 0, 'TEST_gold_reads': 0,
                                         'TEST_answer_reads': 0, 'sealed_rejects_excess_reads': 0, 'provider_response_reads': 0,
                                         'DEV_access': 'denied until model preflight AND full index acceptance pass', 'opaque_integrity_hashing': 'allowed, no decoding'})
    put('entry_audit.json', {'identity': IDENTITY, 'at': utc(), 'B2_protocol_digest': PROTOCOL, 'B2_status': freeze['phase_status'],
                           'corpus40_digest': CORPUS, 'B1_accepted': 75, 'DEV': 20, 'TEST': 55, 'B1_hashes_exact': b1hashes,
                           'B2_completion_SHA256': sha(B2 / 'completion_freeze.json'), 'history_before': verified,
                           'write_root': str(OUT), 'TEST_authorization': 'NOT_GRANTED', 'DEV_semantic_reads': 0})
    return expected


class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in
        ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
         'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage']]


class Budget:
    def __init__(self, prior=None):
        self.limit = read(B2 / 'protocol/compute_budget.json')['future_DEV']
        self.start = time.perf_counter()
        self.counts = dict((prior or {}).get('counts', {}))
        self.peak_RSS = (prior or {}).get('peak_RSS_bytes', 0)
        self.prior_elapsed = (prior or {}).get('elapsed_seconds', 0)
        self.measurement_exception = (prior or {}).get('authorized_measurement_exception')
        self.cpu_start = time.process_time()
        self.last_checkpoint = 0
        self.violation = None
        self.active = True
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.psapi = ctypes.WinDLL('psapi', use_last_error=True)
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while self.active:
            info = MemoryCounters()
            info.cb = ctypes.sizeof(info)
            if not self.psapi.GetProcessMemoryInfo(self.kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
                self.violation = 'MEMORY_SAMPLER_FAILED'
            else:
                self.peak_RSS = max(self.peak_RSS, info.WorkingSetSize, info.PeakWorkingSetSize)
                if self.peak_RSS > self.limit['max_peak_process_RAM_bytes']:
                    self.violation = 'PEAK_PROCESS_RAM_EXCEEDS_10_GIB'
            if self.prior_elapsed + time.perf_counter() - self.start > self.limit['maximum_elapsed_compute_hours'] * 3600:
                self.violation = 'ELAPSED_COMPUTE_LIMIT'
            time.sleep(.1)

    def check(self):
        if self.violation:
            raise RuntimeError(self.violation)
        if time.perf_counter() - self.last_checkpoint >= 30:
            self.checkpoint('RUNNING')

    def checkpoint(self, stage):
        self.last_checkpoint = time.perf_counter()
        put('guards/resource_checkpoint.json', {'at': utc(), 'process_id': os.getpid(), 'stage': stage,
            'counts': dict(self.counts), 'peak_RSS_bytes': self.peak_RSS,
            'elapsed_seconds': self.prior_elapsed + self.last_checkpoint - self.start,
            'current_process_CPU_seconds': time.process_time() - self.cpu_start,
            'violation': self.violation, 'authorized_measurement_exception': self.measurement_exception,
            'final_resource_acceptance': False})

    def charge(self, key, amount=1):
        self.counts[key] = self.counts.get(key, 0) + amount
        if key in self.limit and self.counts[key] > self.limit[key]:
            self.violation = 'COUNT_BUDGET_EXCEEDED:' + key
        self.check()

    def save(self):
        self.active = False
        self.thread.join(timeout=1)
        put('guards/resource_budget.json', {'counts': self.counts, 'peak_RSS_bytes': self.peak_RSS,
            'elapsed_seconds': self.prior_elapsed + time.perf_counter() - self.start, 'violation': self.violation,
            'status': ('PASS_WITH_AUTHORIZED_HISTORICAL_MEASUREMENT_EXCEPTION' if self.measurement_exception else 'PASS') if self.violation is None else 'FAIL', 'limit_source_SHA256': sha(B2 / 'protocol/compute_budget.json'),
            'CPU': True, 'CUDA_inference': 0, 'provider_inference_cost_CNY': 0, 'RSS_sample_interval_ms': 100,
            'authorized_measurement_exception': self.measurement_exception,
            'first_interrupted_final_RAM_CPU': 'NOT_CAPTURED' if self.measurement_exception else 'NOT_APPLICABLE',
            'RAM_acceptance_scope': 'preflight, repaired complete index and DEV; interrupted first index final peak NOT_CAPTURED' if self.measurement_exception else 'all measured processes',
            'guard_enforcement': 'check before and after every bounded batch1 forward; stop on first violation; includes OS peak working set'})
        self.checkpoint('PROCESS_FINALLY_COMPLETE')


def failure_details(exc, stage):
    import traceback
    return {'stage': stage, 'exception_class': type(exc).__name__, 'message_SHA256': hashlib.sha256(str(exc).encode()).hexdigest(),
            'technical_message': str(exc) if stage in ('dependency_preparation', 'package_import', 'model_loading') else None,
            'stack': [{'file': Path(x.filename).name, 'line': x.lineno, 'function': x.name} for x in traceback.extract_tb(exc.__traceback__)],
            'no_model_or_precision_or_contract_switch': True, 'DEV_semantic_reads': 0, 'TEST_semantic_reads': 0}


def close_failure(expected, reason, stage):
    status = 'PHASE4B_V3_B2_D1_MODEL_COMPATIBILITY_NOT_READY'
    boundary = 'READY_FOR_PHASE4B_REASSESSMENT'
    verified = verify_history(expected)
    put('guards/historical_hashes.json', verified)
    put('decision.json', {'identity': IDENTITY, 'phase_status': status, 'completion_boundary': boundary, 'reason': reason,
        'failure_stage': stage, 'DEV_valid_runs': 0, 'DEV_semantic_reads': 0, 'TEST_query_reads': 0, 'TEST_gold_reads': 0, 'TEST_answer_reads': 0,
        'sealed_rejects_excess_reads': 0, 'provider_response_reads': 0, 'index_builds': 0, 'provider_inference_cost_CNY': 0,
        'retrieval_benefit_on_TEST': 'NOT_ASSESSED', 'TEST_authorization': 'NOT_GRANTED', 'history': verified, 'closed_at': utc()})
    report = f'''# PaperResearch V3-B2-D1 Model Preflight

状态：`{status}`；边界：`{boundary}`。冻结模型兼容性未通过，依附件第4/16条停止，未进入native输入读取、索引或DEV。

1. B2 READY及协议digest exact：`{PROTOCOL}`，B2全部artifact SHA256核验一致。
2. B1 accepted75、DEV20、TEST55与construction/TEST seals保持exact，未重划分。
3. 模型权重/tokenizer实际哈希进度：见preflight/model_integrity.json（未完成的项目为NOT_ASSESSED，不能称PASS）。
4. Synthetic真实模型兼容性：未通过；具体阶段`{stage}`，保留preflight/compatibility.json与failure.json，不替换模型/revision/精度。
5. T0完整40索引与chunk数量：NOT_BUILT，未宣称覆盖完成。
6. N1全25762 native units：NOT_BUILT，未静默丢弃单位。
7. M1全1190视觉单位：NOT_BUILT，未降级成N1。
8. 三组检索接口/top5行为实现资格：NOT_ASSESSED（未进入检索执行）。
9. Gold-dependent索引/路由：0，索引运行0；TEST封存访问拒绝PASS。
10. 有效DEV运行次数：0；DEV语义读取0。
11. DEV T0/N1/M1 pageRecall@5：NOT_ASSESSED。
12. TABLE/FIGURE/EQUATION/TEXT分层指标：NOT_ASSESSED。
13. M1−T0 paired DEV difference：NOT_ASSESSED。
14. Native/legacy指标：B2冻结定义与源哈希不变，未以T0 unit=0冒充优势；本阶段评分未执行。
15. 内存/模型准备成本：见guards/resource_budget.json与preflight/acquisition.json；index/query timing与index bytes均未评估。
16. Compute/provider预算：见resource_budget实际记录；provider推理成本0，CPU/float32契约未改变。只尝试允许的冻结依赖/模型准备。
17. 检索配置/索引freeze：未生成qualified configuration；仅封存本次preflight、工程源及失败证据。
18. TEST query/gold/answer、sealed rejects/excess、ROLE_A/B raw responses读取均0。binary完整性hash不解码语义。
19. {len(expected)}个历史文件逐字节SHA256保持，B1/B2/A2与更早证据无变更；所有新增文件位于D1根目录。
20. 最终状态`{status}` / `{boundary}`；TEST收益NOT_ASSESSED，不自动进入B3，不授予TEST解封权限。

技术失败：`{reason}`。完整证据见preflight/failure.json。runtime修复若需突破B2冻结依赖/版本，须另立prospective repair identity；本次没有执行该扩展。
'''
    (OUT / 'report.md').write_text(report, encoding='utf8')
    hashes = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'completion_freeze.json'}
    put('completion_freeze.json', {'identity': IDENTITY, 'phase_status': status, 'completion_boundary': boundary, 'frozen_at': utc(),
        'artifact_sha256_excludes_self': hashes, 'artifact_files_excludes_self': len(hashes),
        'artifact_bytes_excludes_self': sum((OUT / p).stat().st_size for p in hashes), 'automatic_next_stage': False,
        'historical_files_preserved': len(expected), 'DEV_runs': 0, 'TEST_semantic_reads': 0})
    print(json.dumps({'status': status, 'completion_boundary': boundary, 'history_files': len(expected), 'reason': reason}), flush=True)
