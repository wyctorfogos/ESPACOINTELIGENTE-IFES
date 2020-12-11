FROM labviros/is-wire-base:1.1.4-gcc54 as build

ADD conanfile.py .
RUN conan install . -s compiler.libcxx=libstdc++11 --build=missing

ADD . /project
RUN cd /project                                                         \
 && ./build.sh                                                          \
 && mkdir -v -p /tmp/is/lib /tmp/is/bin                                 \
 && libs=`find build/ -type f -name '*.bin' -exec ldd {} \;             \
    | cut -d '(' -f 1 | cut -d '>' -f 2 | sort | uniq`                  \
 && for lib in $libs;                                                   \
    do                                                                  \
      dir="/tmp/is/lib`dirname $lib`";                                  \
      mkdir -v -p  $dir;                                                \
      cp --verbose $lib $dir;                                           \
    done                                                                \
 && cp --verbose `find build/ -type f -name '*.bin'` /tmp/is/bin/


FROM ubuntu:16.04
COPY --from=build /tmp/is/lib/ /
COPY --from=build /tmp/is/bin/ /
COPY --from=build /project/etc/conf/options.json /
