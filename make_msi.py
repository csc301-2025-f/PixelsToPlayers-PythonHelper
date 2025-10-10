# make_msi.py
# Per-user MSI that installs your EXE to %LocalAppData%\<ProductName>,
# registers a custom URL protocol under HKCU\Software\Classes\<protocol>, which is specified in args
# and creates a Desktop shortcut

# TODO: msilib deprecated in python 3.13

import argparse
import uuid
from pathlib import Path
import msilib
from msilib import schema, sequence
from msilib import Directory, Feature, CAB

def guid():
    return "{" + str(uuid.uuid4()).upper() + "}"

def stable_guid(name: str):
    return "{" + str(uuid.uuid5(uuid.NAMESPACE_DNS, name)).upper() + "}"

def add_icon(db, icon_id: str, ico_path: Path):
    msilib.add_data(db, "Icon", [(icon_id, ico_path.read_bytes())])

def build_msi(
        exe_path: Path,
        product_name: str,
        version: str,
        protocol: str,
        out_path: Path,
        manufacturer: str = "My Company",
        icon_path: Path | None = None,
):
    exe_path = exe_path.resolve()
    if not exe_path.exists():
        raise SystemExit(f"EXE not found: {exe_path}")
    if icon_path:
        icon_path = icon_path.resolve()
        if not icon_path.exists():
            raise SystemExit(f"Icon not found: {icon_path}")

    product_code = guid()
    upgrade_code = stable_guid(f"{product_name}-upgradecode")

    # Initialize database + standard sequences
    db = msilib.init_database(str(out_path), schema, product_name, product_code, version, manufacturer)
    msilib.add_tables(db, sequence)

    # Per-user properties + ARP tidying
    props = [
        ("ALLUSERS", "2"),                # enable MSIINSTALLPERUSER semantics
        ("MSIINSTALLPERUSER", "1"),       # force per-user install (Windows Installer 5+)
        ("ARPNOMODIFY", "1"),
        ("ARPNOREPAIR", "1"),
        ("UpgradeCode", upgrade_code),
    ]
    if icon_path:
        add_icon(db, "AppIcon", icon_path)
        props.append(("ARPPRODUCTICON", "AppIcon"))
    msilib.add_data(db, "Property", props)

    # Create a CAB; Directory.add_file() will feed files to this CAB automatically.
    cab = CAB("app.cab")

    # Build Directory tree:
    #   TARGETDIR
    #     └─ LocalAppDataFolder
    #          └─ APPDIR = <ProductName>
    #   DesktopFolder  (for the shortcut)
    srcroot = str(exe_path.parent)  # physical source root for msilib to find files

    root = Directory(db, cab, None, srcroot, "TARGETDIR", "SourceDir")
    localapp = Directory(db, cab, root, srcroot, "LocalAppDataFolder", "LocalAppDataFolder")
    appdir = Directory(db, cab, localapp, srcroot, "APPDIR", product_name)
    desktop = Directory(db, cab, root, srcroot, "DesktopFolder", "DesktopFolder")

    # Main feature; associate with APPDIR
    feature = Feature(db, "MainFeature", product_name, "Core files", 1, directory="APPDIR")

    # Start a component with a stable id and declare the EXE as keypath
    comp_id = f"{appdir.logical}.Component"
    appdir.start_component(comp_id, feature, 0, keyfile=exe_path.name)

    # Add the EXE (to File table + cab)
    file_key = appdir.add_file(exe_path.name, src=str(exe_path))

    # Desktop shortcut (non-advertised): Target = [#<file_key>], WkDir=APPDIR
    msilib.add_data(db, "Shortcut", [(
        "DesktopShortcut",        # Shortcut (primary key)
        "DesktopFolder",          # Directory_
        product_name,             # Name (displayed name)
        comp_id,                  # Component_ (must have a key path)
        f"[#{file_key}]",         # Target (file key reference)
        None,                     # Arguments
        None,                     # Description
        None,                     # Hotkey
        None,                     # Icon_
        None,                     # IconIndex
        None,                     # ShowCmd
        "APPDIR",                 # WkDir
    )])

    # URL protocol registration (HKCU = Root 1)
    cmd = f"\"[#{file_key}]\" \"%1\""
    msilib.add_data(db, "Registry", [
        ("RegBase", 1, fr"Software\\Classes\\{protocol}", None, f"URL:{protocol} Protocol", comp_id),
        ("RegFlag", 1, fr"Software\\Classes\\{protocol}", "URL Protocol", "", comp_id),
        ("RegIcon", 1, fr"Software\\Classes\\{protocol}\\DefaultIcon", None, f"\"[#{file_key}]\",0", comp_id),
        ("RegCmd",  1, fr"Software\\Classes\\{protocol}\\shell\\open\\command", None, cmd, comp_id),
    ])

    # Finalize: write CAB (adds Media row) + commit DB
    cab.commit(db)
    db.Commit()

    print(f"MSI built: {out_path}")
    print(f"Install scope: Per-user → %LocalAppData%\\{product_name}")
    print(f"URL registered: {protocol}:// → {exe_path.name}")
    print("Desktop shortcut: created automatically.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a per-user MSI that installs an EXE, registers a URL protocol, and creates a Desktop shortcut.")
    ap.add_argument("--exe", required=True, help="Path to your packaged EXE.")
    ap.add_argument("--name", required=True, help="Product name (folder & shortcut), e.g., MyApp.")
    ap.add_argument("--version", default="1.0.0", help="Product version x.y.z.")
    ap.add_argument("--protocol", required=True, help="URL scheme, e.g., myapp.")
    ap.add_argument("--out", default="Setup.msi", help="MSI output path.")
    ap.add_argument("--manufacturer", default="My Company", help="Manufacturer shown in Apps & Features.")
    ap.add_argument("--icon", help="Optional .ico for Apps & Features (ARP).")
    args = ap.parse_args()

    build_msi(
        exe_path=Path(args.exe),
        product_name=args.name,
        version=args.version,
        protocol=args.protocol,
        out_path=Path(args.out),
        manufacturer=args.manufacturer,
        icon_path=Path(args.icon) if args.icon else None,
    )
