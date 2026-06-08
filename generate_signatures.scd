// generate_signatures.scd
// Run with: QT_QPA_PLATFORM=offscreen sclang generate_signatures.scd
//
// Dumps all UGen subclass method signatures + extra synthesis-relevant classes
// (Mix, Env, Pan2, etc.) to sc_ugen_signatures.json in the same directory.

(
var file, count, outPath;
var extraClasses, envMethods;

outPath = thisProcess.nowExecutingPath.dirname +/+ "sc_ugen_signatures.json";
file = File(outPath, "w");
file.write("{\n");
count = 0;

// --- 1. All UGen subclasses (.ar, .kr, .ir, .new) ---
// Use findRespondingMethodFor to also find inherited methods
UGen.allSubclasses.do { |cls|
    [\ar, \kr, \ir, \new].do { |selector|
        var m = cls.class.findRespondingMethodFor(selector);
        if(m.notNil) {
            var argNames, key;
            key = cls.name.asString ++ "." ++ selector.asString;
            if(count > 0) { file.write(",\n") };
            argNames = m.argNames.collect(_.asString);
            argNames = argNames[1..]; // skip 'this'
            file.write("  \"" ++ key ++ "\": [");
            argNames.do { |name, i|
                if(i > 0) { file.write(", ") };
                file.write("\"" ++ name ++ "\"");
            };
            file.write("]");
            count = count + 1;
        };
    };
};

// --- 2. Extra non-UGen synthesis classes ---
extraClasses = [
    [Mix, [\ar, \kr, \new, \fill]],
    [Pan2, [\ar, \kr]],
    [Splay, [\ar, \kr]],
    [Balance2, [\ar, \kr]],
    [LinPan2, [\ar, \kr]],
    [Rotate2, [\ar, \kr]],
    [PanAz, [\ar, \kr]],
    [PanB, [\ar, \kr]],
    [PanB2, [\ar, \kr]],
    [DecodeB2, [\ar, \kr]],
    [BiPanB2, [\ar, \kr]],
    [LinXFade2, [\ar, \kr]],
    [XFade2, [\ar, \kr]],
    [Select, [\ar, \kr]],
    [SelectX, [\ar, \kr]],
    [DemandEnvGen, [\ar, \kr]],
    [Klang, [\ar]],
    [Klank, [\ar]],
    [DynKlang, [\ar, \kr]],
    [DynKlank, [\ar, \kr]],
];

extraClasses.do { |pair|
    var cls = pair[0];
    var selectors = pair[1];
    selectors.do { |selector|
        var m = cls.class.findRespondingMethodFor(selector);
        if(m.notNil) {
            var argNames, key;
            key = cls.name ++ "." ++ selector;
            argNames = m.argNames.collect(_.asString);
            argNames = argNames[1..]; // skip 'this'
            if(count > 0) { file.write(",\n") };
            file.write("  \"" ++ key ++ "\": [");
            argNames.do { |name, i|
                if(i > 0) { file.write(", ") };
                file.write("\"" ++ name ++ "\"");
            };
            file.write("]");
            count = count + 1;
        };
    };
};

// --- 3. Env convenience methods ---
envMethods = [\new, \perc, \linen, \adsr, \asr, \dadsr, \sine, \triangle, \cutoff, \circle, \step, \pairs];

envMethods.do { |selector|
    var m = Env.class.findRespondingMethodFor(selector);
    if(m.notNil) {
        var argNames, key;
        key = "Env." ++ selector;
        argNames = m.argNames.collect(_.asString);
        argNames = argNames[1..]; // skip 'this'
        if(count > 0) { file.write(",\n") };
        file.write("  \"" ++ key ++ "\": [");
        argNames.do { |name, i|
            if(i > 0) { file.write(", ") };
            file.write("\"" ++ name ++ "\"");
        };
        file.write("]");
        count = count + 1;
    };
};

file.write("\n}\n");
file.close;

("Dumped " ++ count ++ " method signatures to: " ++ outPath).postln;
0.exit;
)
