# -*- mode: python -*-

from PyInstaller.utils import hooks
block_cipher = None

datas = []
datas += [('styles/dark.qss', 'styles')]
datas += [('styles/rc/*.png', 'styles/rc')]

a = Analysis(['gui.py'],
             pathex=['/home/ayyerkar/repos/CLEMGui'],
             binaries=[],
             datas=datas,
             hiddenimports=['pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='Clement',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
