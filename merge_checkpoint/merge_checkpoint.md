# Merge checkpoint.pth Split Files
## Windows Merge Command (PowerShell):
cd [directory where split files are located]
copy /b checkpoint_part_* checkpoint.pth

## Linux/Mac Merge Command:
cat checkpoint_part_* > checkpoint.pth

## Verify Integrity (Optional)
# Windows
Get-FileHash -Algorithm MD5 checkpoint.pth
# Linux/Mac
md5sum checkpoint.pth
# Original File MD5: 34f6adb4fb5217d9b432d4f567b513fe