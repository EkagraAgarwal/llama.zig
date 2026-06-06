const std = @import("std");
const builtin = @import("builtin");

pub const MappedFile = struct {
    data: []align(4096) const u8,
    file_handle: if (builtin.os.tag == .windows) std.os.windows.HANDLE else void,
    mapping_handle: if (builtin.os.tag == .windows) std.os.windows.HANDLE else void,

    pub fn init(path: []const u8) !MappedFile {
        if (builtin.os.tag != .windows) {
            @compileError("mmap not yet implemented for this OS");
        }

        const windows = std.os.windows;

        // Constants
        const GENERIC_READ = 0x80000000;
        const FILE_SHARE_READ = 1;
        const OPEN_EXISTING = 3;
        const FILE_ATTRIBUTE_NORMAL = 0x80;
        const FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000;
        const PAGE_READONLY = 0x02;
        const FILE_MAP_READ = 4;
        const INVALID_HANDLE_VALUE = @as(windows.HANDLE, @ptrFromInt(@as(usize, @bitCast(@as(isize, -1)))));

        // Path conversion to UTF-16 null-terminated for Windows APIs
        var path_w: [32768:0]u16 = undefined;
        const len = try std.unicode.utf8ToUtf16Le(&path_w, path);
        path_w[len] = 0;

        const file_h = CreateFileW(
            &path_w,
            GENERIC_READ,
            FILE_SHARE_READ,
            null,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
            null,
        );

        if (file_h == INVALID_HANDLE_VALUE) {
            const err = GetLastError();
            return switch (err) {
                2 => error.FileNotFound, // ERROR_FILE_NOT_FOUND
                3 => error.PathNotFound, // ERROR_PATH_NOT_FOUND
                5 => error.AccessDenied, // ERROR_ACCESS_DENIED
                else => error.SystemResources,
            };
        }
        errdefer _ = CloseHandle(file_h);

        var file_size: i64 = 0;
        if (GetFileSizeEx(file_h, &file_size) == 0) {
            return error.SystemResources;
        }

        if (file_size == 0) {
            return error.EmptyFile;
        }

        const size = @as(u64, @intCast(file_size));

        const mapping_h = CreateFileMappingW(
            file_h,
            null,
            PAGE_READONLY,
            0,
            0,
            null,
        ) orelse {
            const err = GetLastError();
            return switch (err) {
                5 => error.AccessDenied,
                else => error.SystemResources,
            };
        };
        errdefer _ = CloseHandle(mapping_h);

        const view_ptr = MapViewOfFile(
            mapping_h,
            FILE_MAP_READ,
            0,
            0,
            0,
        ) orelse {
            return error.SystemResources;
        };

        const slice_ptr = @as([*]const u8, @ptrCast(view_ptr));
        const data = slice_ptr[0..size];

        return MappedFile{
            .data = @alignCast(data),
            .file_handle = file_h,
            .mapping_handle = mapping_h,
        };
    }

    pub fn deinit(self: *MappedFile) void {
        if (builtin.os.tag == .windows) {
            _ = UnmapViewOfFile(self.data.ptr);
            _ = CloseHandle(self.mapping_handle);
            _ = CloseHandle(self.file_handle);
        }
    }

    pub fn slice(self: *const MappedFile, offset: u64, len: u64) ![]const u8 {
        const end = std.math.add(u64, offset, len) catch return error.OutOfBounds;
        if (end > self.data.len) return error.OutOfBounds;
        return self.data[offset..end];
    }
};

// Win32 Extern Declarations
extern "kernel32" fn CreateFileW(
    lpFileName: [*:0]const u16,
    dwDesiredAccess: u32,
    dwShareMode: u32,
    lpSecurityAttributes: ?*anyopaque,
    dwCreationDisposition: u32,
    dwFlagsAndAttributes: u32,
    hTemplateFile: ?*anyopaque,
) callconv(.winapi) std.os.windows.HANDLE;

extern "kernel32" fn GetFileSizeEx(
    hFile: std.os.windows.HANDLE,
    lpFileSize: *i64,
) callconv(.winapi) i32;

extern "kernel32" fn CreateFileMappingW(
    hFile: std.os.windows.HANDLE,
    lpFileMappingAttributes: ?*anyopaque,
    flProtect: u32,
    dwMaximumSizeHigh: u32,
    dwMaximumSizeLow: u32,
    lpName: ?[*:0]const u16,
) callconv(.winapi) ?std.os.windows.HANDLE;

extern "kernel32" fn MapViewOfFile(
    hFileMappingObject: std.os.windows.HANDLE,
    dwDesiredAccess: u32,
    dwFileOffsetHigh: u32,
    dwFileOffsetLow: u32,
    dwNumberOfBytesToMap: usize,
) callconv(.winapi) ?*anyopaque;

extern "kernel32" fn UnmapViewOfFile(
    lpBaseAddress: ?*const anyopaque,
) callconv(.winapi) i32;

extern "kernel32" fn CloseHandle(
    hObject: std.os.windows.HANDLE,
) callconv(.winapi) i32;

extern "kernel32" fn GetLastError() callconv(.winapi) u32;
