# Audio Analysis & Dedupe

一个 MCP 服务器：检查音频文件夹 —— **这些文件里是什么**，以及**哪些是重复的**。

[English](README.md) · 许可证：[AGPL-3.0](LICENSE)

## 功能

- **资料库分析**：逐个文件给出大小、时长、采样率、声道数、编码与码率（支持递归）。
- **完全重复**：按 SHA-256 分组字节级相同的文件，并给出建议保留哪个。
- **同一音频不同容器**：解码后的 PCM 数据相同（例如同一母带导出成 WAV/AIFF/AU）会单独归类。
- **近似重复候选**：比较文件名（自动忽略 `(1)`、`- copy`、`（1）` 这类后缀）、时长与大小，
  并列出每对候选的具体差异。
- **只读设计**：不删除、不移动、不重命名、不改写文件，也不碰任何厂商数据库。

## 环境要求

| | |
| --- | --- |
| 系统 | macOS / Linux / Windows |
| Python | 3.9 或更新 |
| 可选 | `PATH` 里有 `ffprobe`（或设置 `SHADOW_FFPROBE`），用于 WAV/AIFF/AU 之外的格式 |

## 安装

```sh
# 作为 Codex 插件
codex plugin marketplace add shadowroommusic/audio-analysis-dedupe
codex plugin add audio-analysis-dedupe@shadowroom

# 只用命令行
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/shadow-audio-dedupe --help
```

MCP 客户端配置：

```json
{
  "mcpServers": {
    "audio-analysis-dedupe": {
      "command": "python3",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/audio-analysis-dedupe"
    }
  }
}
```

## 配置项

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `SHADOW_FFPROBE` | 自动检测 | 非 PCM 容器使用的 `ffprobe` 路径 |
| `--threshold` | `0.72` | 近似重复的相似度阈值 |
| `--output` | stdout | 把 JSON 报告写到文件 |

## 工具

| 工具 | 作用 |
| --- | --- |
| `analyze_file` | 分析单个文件（格式、时长、采样率、声道、码率） |
| `dedupe_folder` | 分析文件夹并按重复分组（`recursive`、`threshold`、`max_candidates`） |

命令行等价：`shadow-audio-dedupe analyze --path <文件或文件夹>`、`shadow-audio-dedupe dedupe --folder <文件夹>`。

## 用法

```sh
# 这个 crate 里都有什么？
.venv/bin/shadow-audio-dedupe analyze --path ~/Music/Crate --output analysis.json

# 哪些重复？各版本差在哪？
.venv/bin/shadow-audio-dedupe dedupe --folder ~/Music/Crate --threshold 0.72 --output duplicates.json
```

报告包含 `exact_groups`、`identical_audio_groups`、`candidates`（含 `reasons` 与 `differences`）
以及带冗余文件数与冗余字节数的 `summary`。

## 安全说明

- 全程只读；不会打开任何厂商数据库。
- 报告只写到你用 `--output` 指定的路径（或 stdout）。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| MP3/FLAC 读不到元数据 | 安装 `ffmpeg`/`ffprobe`，或用 `SHADOW_FFPROBE` 指定路径。 |
| 全都算成近似重复 | 调高 `--threshold`（例如 `0.85`）。 |
| 文件夹太大很慢 | 缩小范围，或先 `analyze` 看看里面有什么。 |

## 参与开发

见 [CONTRIBUTING.md](CONTRIBUTING.md)；实现细节在 [docs/internals.md](docs/internals.md)。

## 许可证

AGPL-3.0，见 [LICENSE](LICENSE)。可选的 `ffprobe` 使用受本机 FFmpeg 安装的许可证约束。
