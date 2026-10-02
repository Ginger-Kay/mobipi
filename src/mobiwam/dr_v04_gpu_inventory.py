"""Read-only NVML compute and graphics context inventory, including hidden host PIDs."""
import ctypes


def inventory():
    nv=ctypes.CDLL('libnvidia-ml.so.1')
    if nv.nvmlInit_v2()!=0:raise RuntimeError('NVML init failed')
    class Info(ctypes.Structure):
        _fields_=[('pid',ctypes.c_uint),('usedGpuMemory',ctypes.c_ulonglong),('gpuInstanceId',ctypes.c_uint),('computeInstanceId',ctypes.c_uint)]
    rows=[]
    try:
        count=ctypes.c_uint();assert nv.nvmlDeviceGetCount_v2(ctypes.byref(count))==0
        for i in range(count.value):
            d=ctypes.c_void_p();assert nv.nvmlDeviceGetHandleByIndex_v2(i,ctypes.byref(d))==0
            uuid=ctypes.create_string_buffer(96);assert nv.nvmlDeviceGetUUID(d,uuid,96)==0
            row=dict(index=i,uuid=uuid.value.decode())
            for kind in ('Compute','Graphics'):
                size=ctypes.c_uint(128);arr=(Info*128)();rc=getattr(nv,'nvmlDeviceGet'+kind+'RunningProcesses_v3')(d,ctypes.byref(size),arr)
                if rc:raise RuntimeError('NVML process inventory incomplete: '+str(rc))
                row[kind.lower()]=[dict(pid=x.pid,memory_MiB=x.usedGpuMemory/1024**2) for x in arr[:size.value]]
            rows.append(row)
    finally:nv.nvmlShutdown()
    return rows


def require_occupancy_only(rows,expected):
    if len(rows)!=4:raise ValueError('GPU allocation changed')
    for row in rows:
        saved=expected[str(row['index'])]
        if row['uuid']!=saved['uuid'] or {x['pid'] for x in row['compute']}!={saved['occupancy_pid']} or row['graphics']:
            raise ValueError('unknown/other GPU workload; preserve it, do not launch')
