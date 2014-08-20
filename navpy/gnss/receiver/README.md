# Receiver Class

Navpy GNSS's `receiver` class holds a snapshot of data held inside a GNSS receiver. 
It is a snapshot because it only holds data from a *single* measurement epoch, **not**
a time history of measurements (i.e. multiple epochs). 

The class is structured as follows. 
*	In the top layers the position, velocity, and time (PVT) of the receiver.
*	Nested under the `rawData` class is a set of APIs that can set or get raw satellite
	measurements like pseudorange and carrier phase (or accumulated delta range).

The tree diagram of the `receiver` class structure is shown below.
`
Receiver
|
|- TOW
|- lat
|- lon
|- alt
|- clkbias
|- vN
|- vE
|- vD
|- sig_N
|- sig_E
|- sig_D
|- rawdata
	|- set_pseudorange()
	|- set_PR_cov()
	|- set_carrierphase()
	|- set_ADR_cov()
	|- set_doppler()
	|- set_CNo()
	|- check_dataValid()
	|- get_pseudorange()
	|- get_PR_cov()
	|- get_carrierphase()
	|- get_ADR_cov()
	|- get_doppler()
	|- get_CNo()
	|- is_dataValid()
`