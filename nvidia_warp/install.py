# omni.kit.pipapi extension is required
import omni.kit.pipapi

omni.kit.pipapi.install(
    package="semver",
    version="2.13.0",
    module="semver", # sometimes module is different from package name, module is used for import check
    ignore_import_check=False,
    ignore_cache=False,
    use_online_index=True,
    surpress_output=False,
    extra_args=[]
)

# use the newly installed package
import semver
ver = semver.VersionInfo.parse('1.2.3-pre.2+build.4')
print(ver)
