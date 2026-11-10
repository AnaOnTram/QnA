/* Persistent HTP scorer for the Q6A ResNet-50 context.
 *
 * Loads libQnnHtp.so + the .bin once, then scores native uint8 tensors from
 * stdin until EOF. One process == one FastRPC session, which is what keeps
 * the QCS6490 CDSP from hanging (qcom,fastrpc unmmap / glink intent timeout
 * after a handful of qnn-net-run process lifetimes).
 *
 * Protocol, all little-endian:
 *   stdout first line: READY <in_bytes> <out_bytes> <graph>\n
 *   then repeat:
 *     stdin:  uint32 n, n * in_bytes
 *     stdout: uint32 n, n * out_bytes
 *
 * Logs go to stderr. Compile on the board (see install command in npu_remote.py).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include "QnnInterface.h"
#include "QnnLog.h"
#include "System/QnnSystemInterface.h"
#include "QnnHtpDevice.h"

#define DIE(...) do { fprintf(stderr, "npu-scorer: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)
#define CHK(expr, what) do { \
    Qnn_ErrorHandle_t _e = (expr); \
    if (_e != QNN_SUCCESS) DIE("%s failed: 0x%llx", what, (unsigned long long)_e); \
  } while (0)

static QNN_INTERFACE_VER_TYPE qnn;
static QNN_SYSTEM_INTERFACE_VER_TYPE qnn_sys;
static int have_sys;

static void log_cb(const char *fmt, QnnLog_Level_t level, uint64_t ts, va_list args) {
  const char *lvl = level == QNN_LOG_LEVEL_ERROR ? "E" :
                    level == QNN_LOG_LEVEL_WARN  ? "W" :
                    level == QNN_LOG_LEVEL_INFO  ? "I" : "V";
  fprintf(stderr, "[qnn %s %llu] ", lvl, (unsigned long long)ts);
  vfprintf(stderr, fmt, args);
  if (fmt && fmt[0] && fmt[strlen(fmt) - 1] != '\n') fputc('\n', stderr);
}

static void *must_dlopen(const char *path) {
  void *h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
  if (!h) DIE("dlopen %s: %s", path, dlerror());
  return h;
}

static int read_full(int fd, void *buf, size_t n) {
  uint8_t *p = buf;
  while (n) {
    ssize_t r = read(fd, p, n);
    if (r == 0) return 1;          /* EOF */
    if (r < 0) { if (errno == EINTR) continue; return -1; }
    p += (size_t)r;
    n -= (size_t)r;
  }
  return 0;
}

static int write_full(int fd, const void *buf, size_t n) {
  const uint8_t *p = buf;
  while (n) {
    ssize_t w = write(fd, p, n);
    if (w < 0) { if (errno == EINTR) continue; return -1; }
    p += (size_t)w;
    n -= (size_t)w;
  }
  return 0;
}

static uint32_t dtype_bytes(Qnn_DataType_t dt) {
  switch (dt) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_FLOAT_64:
      return 8;
    default:
      return 1;
  }
}

static uint32_t tensor_rank(const Qnn_Tensor_t *t) {
  return t->version == QNN_TENSOR_VERSION_2 ? t->v2.rank : t->v1.rank;
}

static uint32_t *tensor_dims(const Qnn_Tensor_t *t) {
  return t->version == QNN_TENSOR_VERSION_2 ? t->v2.dimensions : t->v1.dimensions;
}

static Qnn_DataType_t tensor_dtype(const Qnn_Tensor_t *t) {
  return t->version == QNN_TENSOR_VERSION_2 ? t->v2.dataType : t->v1.dataType;
}

static uint32_t tensor_nbytes(const Qnn_Tensor_t *t) {
  uint32_t rank = tensor_rank(t);
  uint32_t *dims = tensor_dims(t);
  uint32_t n = dtype_bytes(tensor_dtype(t));
  for (uint32_t i = 0; i < rank; i++) n *= dims[i] ? dims[i] : 1;
  return n;
}

static void tensor_set_buf(Qnn_Tensor_t *t, void *data, uint32_t n) {
  if (t->version == QNN_TENSOR_VERSION_2) {
    t->v2.memType = QNN_TENSORMEMTYPE_RAW;
    t->v2.clientBuf.data = data;
    t->v2.clientBuf.dataSize = n;
  } else {
    t->v1.memType = QNN_TENSORMEMTYPE_RAW;
    t->v1.clientBuf.data = data;
    t->v1.clientBuf.dataSize = n;
  }
}

static void copy_tensor(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src) {
  *dst = *src;
  if (dst->version == QNN_TENSOR_VERSION_2 && src->v2.name)
    dst->v2.name = strdup(src->v2.name);
  else if (dst->version == QNN_TENSOR_VERSION_1 && src->v1.name)
    dst->v1.name = strdup(src->v1.name);
  uint32_t rank = tensor_rank(src);
  uint32_t *odims = tensor_dims(src);
  if (rank && odims) {
    uint32_t *d = calloc(rank, sizeof(uint32_t));
    if (!d) DIE("oom dims");
    memcpy(d, odims, rank * sizeof(uint32_t));
    if (dst->version == QNN_TENSOR_VERSION_2) dst->v2.dimensions = d;
    else dst->v1.dimensions = d;
  }
  if (dst->version == QNN_TENSOR_VERSION_2 && src->v2.isDynamicDimensions && rank) {
    uint8_t *b = calloc(rank, 1);
    if (!b) DIE("oom dyn");
    memcpy(b, src->v2.isDynamicDimensions, rank);
    dst->v2.isDynamicDimensions = b;
  }
}

static void make_hardcoded(Qnn_Tensor_t *in, Qnn_Tensor_t *out,
                           uint32_t **keep_in, uint32_t **keep_out,
                           uint32_t in_id, uint32_t out_id) {
  static uint32_t in_dims[4] = {1, 3, 224, 224};
  static uint32_t out_dims[2] = {1, 1000};
  *in = (Qnn_Tensor_t)QNN_TENSOR_INIT;
  in->version = QNN_TENSOR_VERSION_1;
  in->v1.id = in_id;
  in->v1.name = "input";
  in->v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  in->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  in->v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
  in->v1.rank = 4;
  in->v1.dimensions = in_dims;
  in->v1.memType = QNN_TENSORMEMTYPE_RAW;
  *out = (Qnn_Tensor_t)QNN_TENSOR_INIT;
  out->version = QNN_TENSOR_VERSION_1;
  out->v1.id = out_id;
  out->v1.name = "output";
  out->v1.type = QNN_TENSOR_TYPE_APP_READ;
  out->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  out->v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
  out->v1.rank = 2;
  out->v1.dimensions = out_dims;
  out->v1.memType = QNN_TENSORMEMTYPE_RAW;
  *keep_in = in_dims;
  *keep_out = out_dims;
}

static void load_interfaces(const char *sys_path) {
  typedef Qnn_ErrorHandle_t (*GetProv)(const QnnInterface_t ***, uint32_t *);
  typedef Qnn_ErrorHandle_t (*GetSys)(const QnnSystemInterface_t ***, uint32_t *);
  void *htp = must_dlopen("libQnnHtp.so");
  GetProv get = (GetProv)dlsym(htp, "QnnInterface_getProviders");
  if (!get) DIE("QnnInterface_getProviders missing");
  const QnnInterface_t **prov = NULL;
  uint32_t n = 0;
  CHK(get(&prov, &n), "getProviders");
  if (!n || !prov || !prov[0]) DIE("no QNN providers");
  qnn = prov[0]->QNN_INTERFACE_VER_NAME;
  fprintf(stderr, "npu-scorer: HTP core API %u.%u.%u backend %u.%u.%u provider=%s\n",
          prov[0]->apiVersion.coreApiVersion.major,
          prov[0]->apiVersion.coreApiVersion.minor,
          prov[0]->apiVersion.coreApiVersion.patch,
          prov[0]->apiVersion.backendApiVersion.major,
          prov[0]->apiVersion.backendApiVersion.minor,
          prov[0]->apiVersion.backendApiVersion.patch,
          prov[0]->providerName ? prov[0]->providerName : "?");

  void *sys = dlopen(sys_path, RTLD_NOW | RTLD_GLOBAL);
  if (!sys) sys = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
  if (!sys) {
    fprintf(stderr, "npu-scorer: libQnnSystem.so not loaded (%s); using hardcoded I/O\n", dlerror());
    return;
  }
  GetSys gets = (GetSys)dlsym(sys, "QnnSystemInterface_getProviders");
  if (!gets) {
    fprintf(stderr, "npu-scorer: QnnSystemInterface_getProviders missing\n");
    return;
  }
  const QnnSystemInterface_t **sp = NULL;
  uint32_t ns = 0;
  if (gets(&sp, &ns) != QNN_SUCCESS || !ns || !sp || !sp[0]) {
    fprintf(stderr, "npu-scorer: no system providers\n");
    return;
  }
  qnn_sys = sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
  have_sys = 1;
}

static int parse_binary_io(void *blob, uint64_t blob_sz, const char *want_graph,
                           Qnn_Tensor_t *in, Qnn_Tensor_t *out, char **graph_name) {
  if (!have_sys || !qnn_sys.systemContextCreate || !qnn_sys.systemContextGetBinaryInfo)
    return 0;
  QnnSystemContext_Handle_t sys = NULL;
  if (qnn_sys.systemContextCreate(&sys) != QNN_SUCCESS) return 0;
  const QnnSystemContext_BinaryInfo_t *info = NULL;
  Qnn_ContextBinarySize_t info_sz = 0;
  Qnn_ErrorHandle_t e = qnn_sys.systemContextGetBinaryInfo(sys, blob, blob_sz, &info, &info_sz);
  if (e != QNN_SUCCESS || !info) {
    fprintf(stderr, "npu-scorer: getBinaryInfo 0x%llx (will hardcode I/O)\n", (unsigned long long)e);
    if (qnn_sys.systemContextFree) qnn_sys.systemContextFree(sys);
    return 0;
  }

  uint32_t ngraphs = 0;
  QnnSystemContext_GraphInfo_t *graphs = NULL;
  if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    ngraphs = info->contextBinaryInfoV1.numGraphs;
    graphs = info->contextBinaryInfoV1.graphs;
  } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    ngraphs = info->contextBinaryInfoV2.numGraphs;
    graphs = info->contextBinaryInfoV2.graphs;
  } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    ngraphs = info->contextBinaryInfoV3.numGraphs;
    graphs = info->contextBinaryInfoV3.graphs;
  } else {
    fprintf(stderr, "npu-scorer: unknown binary info version %u\n", info->version);
    qnn_sys.systemContextFree(sys);
    return 0;
  }
  if (!ngraphs || !graphs) {
    qnn_sys.systemContextFree(sys);
    return 0;
  }

  QnnSystemContext_GraphInfo_t *g = &graphs[0];
  for (uint32_t i = 0; i < ngraphs; i++) {
    const char *nm = graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3
                         ? graphs[i].graphInfoV3.graphName
                         : graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2
                         ? graphs[i].graphInfoV2.graphName
                         : graphs[i].graphInfoV1.graphName;
    if (want_graph && nm && strcmp(nm, want_graph) == 0) { g = &graphs[i]; break; }
  }

  const char *nm;
  uint32_t nIn, nOut;
  Qnn_Tensor_t *ins, *outs;
  if (g->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
    nm = g->graphInfoV3.graphName;
    nIn = g->graphInfoV3.numGraphInputs;
    ins = g->graphInfoV3.graphInputs;
    nOut = g->graphInfoV3.numGraphOutputs;
    outs = g->graphInfoV3.graphOutputs;
  } else if (g->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
    nm = g->graphInfoV2.graphName;
    nIn = g->graphInfoV2.numGraphInputs;
    ins = g->graphInfoV2.graphInputs;
    nOut = g->graphInfoV2.numGraphOutputs;
    outs = g->graphInfoV2.graphOutputs;
  } else {
    nm = g->graphInfoV1.graphName;
    nIn = g->graphInfoV1.numGraphInputs;
    ins = g->graphInfoV1.graphInputs;
    nOut = g->graphInfoV1.numGraphOutputs;
    outs = g->graphInfoV1.graphOutputs;
  }
  if (nIn != 1 || nOut != 1 || !ins || !outs) {
    fprintf(stderr, "npu-scorer: graph %s has %u in / %u out; expected 1/1\n",
            nm ? nm : "?", nIn, nOut);
    qnn_sys.systemContextFree(sys);
    return 0;
  }
  copy_tensor(in, &ins[0]);
  copy_tensor(out, &outs[0]);
  *graph_name = strdup(nm ? nm : want_graph);
  fprintf(stderr, "npu-scorer: graph %s in=%uB out=%uB (from binary metadata)\n",
          *graph_name, tensor_nbytes(in), tensor_nbytes(out));
  qnn_sys.systemContextFree(sys);
  return 1;
}

static Qnn_DeviceHandle_t make_device(Qnn_LogHandle_t logger) {
  if (!qnn.deviceCreate) return NULL;
  QnnHtpDevice_CustomConfig_t soc = {
      .option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC, .socModel = 498};
  QnnHtpDevice_CustomConfig_t arch = {
      .option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH,
      .arch = {.deviceId = 0, .arch = QNN_HTP_DEVICE_ARCH_V68}};
  QnnHtpDevice_CustomConfig_t pd = {
      .option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD,
      .useSignedProcessDomain = {.deviceId = 0, .useSignedProcessDomain = false}};
  QnnDevice_Config_t c0 = {.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM, .customConfig = &soc};
  QnnDevice_Config_t c1 = {.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM, .customConfig = &arch};
  QnnDevice_Config_t c2 = {.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM, .customConfig = &pd};
  const QnnDevice_Config_t *cfgs[] = {&c0, &c1, &c2, NULL};
  Qnn_DeviceHandle_t dev = NULL;
  Qnn_ErrorHandle_t e = qnn.deviceCreate(logger, cfgs, &dev);
  if (e == QNN_SUCCESS) return dev;
  fprintf(stderr, "npu-scorer: deviceCreate(configured) 0x%llx, retrying default\n",
          (unsigned long long)e);
  dev = NULL;
  e = qnn.deviceCreate(logger, NULL, &dev);
  if (e != QNN_SUCCESS) {
    fprintf(stderr, "npu-scorer: deviceCreate(default) 0x%llx, continuing with NULL device\n",
            (unsigned long long)e);
    return NULL;
  }
  return dev;
}

int main(int argc, char **argv) {
  const char *bin_path = argc > 1 ? argv[1] : "resnet50_aimet_quantized_6490.bin";
  const char *want_graph = argc > 2 ? argv[2] : "resnet50_aimet";
  const char *sys_path = argc > 3 ? argv[3] : "libQnnSystem.so";
  /* Tensor ids differ per export (V1 output is 399, V2 is 454). Binary info
     parsing is unavailable for newer context binaries when compiled against
     older headers, so the ids are supplied by the caller. */
  uint32_t in_id = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 0) : 1;
  uint32_t out_id = argc > 5 ? (uint32_t)strtoul(argv[5], NULL, 0) : 399;

  setvbuf(stdin, NULL, _IONBF, 0);
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  /* libQnnHtp prints rpcmem dummy lines on stdout. Park stdout on stderr
     until the binary protocol starts, otherwise the host cannot parse READY. */
  int saved_out = dup(STDOUT_FILENO);
  if (saved_out < 0) DIE("dup stdout: %s", strerror(errno));
  if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) DIE("dup2 stderr: %s", strerror(errno));

  load_interfaces(sys_path);

  int fd = open(bin_path, O_RDONLY);
  if (fd < 0) DIE("open %s: %s", bin_path, strerror(errno));
  struct stat st;
  if (fstat(fd, &st) < 0) DIE("fstat: %s", strerror(errno));
  void *blob = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (blob == MAP_FAILED) DIE("mmap: %s", strerror(errno));

  Qnn_Tensor_t in_t, out_t;
  uint32_t *keep_in = NULL, *keep_out = NULL;
  char *graph_name = NULL;
  if (!parse_binary_io(blob, (uint64_t)st.st_size, want_graph, &in_t, &out_t, &graph_name)) {
    make_hardcoded(&in_t, &out_t, &keep_in, &keep_out, in_id, out_id);
    graph_name = strdup(want_graph);
    fprintf(stderr, "npu-scorer: hardcoded graph %s in=%uB out=%uB\n",
            graph_name, tensor_nbytes(&in_t), tensor_nbytes(&out_t));
  }

  fprintf(stderr,
          "npu-scorer: fn logCreate=%p backendCreate=%p deviceCreate=%p "
          "contextCreateFromBinary=%p graphRetrieve=%p graphExecute=%p\n",
          (void *)qnn.logCreate, (void *)qnn.backendCreate, (void *)qnn.deviceCreate,
          (void *)qnn.contextCreateFromBinary, (void *)qnn.graphRetrieve, (void *)qnn.graphExecute);
  if (!qnn.logCreate || !qnn.backendCreate || !qnn.contextCreateFromBinary ||
      !qnn.graphRetrieve || !qnn.graphExecute)
    DIE("required QNN function pointer is NULL (header/lib version skew)");

  Qnn_LogHandle_t logger = NULL;
  fprintf(stderr, "npu-scorer: logCreate...\n");
  CHK(qnn.logCreate(log_cb, QNN_LOG_LEVEL_ERROR, &logger), "logCreate");

  Qnn_BackendHandle_t backend = NULL;
  fprintf(stderr, "npu-scorer: backendCreate...\n");
  CHK(qnn.backendCreate(logger, NULL, &backend), "backendCreate");

  /* NULL device: HTP picks the default CDSP. Custom QnnHtpDevice_Config
     structs are version-skew-sensitive and are not required for this graph. */
  Qnn_DeviceHandle_t device = NULL;
  (void)make_device;

  Qnn_ContextHandle_t ctx = NULL;
  fprintf(stderr, "npu-scorer: contextCreateFromBinary (%lld bytes)...\n",
          (long long)st.st_size);
  CHK(qnn.contextCreateFromBinary(backend, device, NULL, blob, (Qnn_ContextBinarySize_t)st.st_size,
                                  &ctx, NULL),
      "contextCreateFromBinary");

  Qnn_GraphHandle_t graph = NULL;
  fprintf(stderr, "npu-scorer: graphRetrieve %s...\n", graph_name);
  CHK(qnn.graphRetrieve(ctx, graph_name, &graph), "graphRetrieve");

  uint32_t in_n = tensor_nbytes(&in_t);
  uint32_t out_n = tensor_nbytes(&out_t);
  void *in_buf = calloc(1, in_n);
  void *out_buf = calloc(1, out_n);
  if (!in_buf || !out_buf) DIE("oom io buffers");
  tensor_set_buf(&in_t, in_buf, in_n);
  tensor_set_buf(&out_t, out_buf, out_n);

  fprintf(stderr, "npu-scorer: context loaded, waiting on stdin\n");
  fflush(stderr);
  if (dup2(saved_out, STDOUT_FILENO) < 0) DIE("restore stdout: %s", strerror(errno));
  close(saved_out);
  saved_out = -1;
  printf("READY %u %u %s\n", in_n, out_n, graph_name);
  fflush(stdout);

  /* Optional board-local benchmark: excludes transport, input encoding and file I/O. */
  if (argc > 7) {
    FILE *sample = fopen(argv[6], "rb");
    if (!sample || fread(in_buf, 1, in_n, sample) != in_n) DIE("read benchmark sample");
    fclose(sample);
    int runs = atoi(argv[7]);
    if (runs < 1 || runs > 100000) DIE("bad benchmark count");
    for (int i = 0; i < 20; i++)
      CHK(qnn.graphExecute(graph, &in_t, 1, &out_t, 1, NULL, NULL), "warmup");
    printf("{\"warmups\":20,\"samples_ms\":[");
    for (int i = 0; i < runs; i++) {
      struct timespec a, b;
      clock_gettime(CLOCK_MONOTONIC, &a);
      CHK(qnn.graphExecute(graph, &in_t, 1, &out_t, 1, NULL, NULL), "benchmark execute");
      clock_gettime(CLOCK_MONOTONIC, &b);
      double ms = (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
      printf("%s%.9f", i ? "," : "", ms);
    }
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("],\"peak_rss_kib\":%ld}\n", usage.ru_maxrss);
    goto cleanup;
  }

  for (;;) {
    uint32_t n = 0;
    int rr = read_full(STDIN_FILENO, &n, 4);
    if (rr == 1) break;
    if (rr < 0) DIE("read n: %s", strerror(errno));
    if (n == 0 || n > 4096) DIE("bad batch %u", n);
    for (uint32_t i = 0; i < n; i++) {
      if (read_full(STDIN_FILENO, in_buf, in_n) != 0) DIE("read input %u failed", i);
      CHK(qnn.graphExecute(graph, &in_t, 1, &out_t, 1, NULL, NULL), "graphExecute");
      if (i == 0) {
        if (write_full(STDOUT_FILENO, &n, 4) < 0) DIE("write n");
      }
      if (write_full(STDOUT_FILENO, out_buf, out_n) < 0) DIE("write output %u", i);
    }
  }

cleanup:
  if (qnn.contextFree) qnn.contextFree(ctx, NULL);
  if (qnn.backendFree) qnn.backendFree(backend);
  if (device && qnn.deviceFree) qnn.deviceFree(device);
  if (logger && qnn.logFree) qnn.logFree(logger);
  return 0;
}
