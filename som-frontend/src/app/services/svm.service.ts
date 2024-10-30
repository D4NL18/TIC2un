import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SvmService {

  private apiUrl = 'http://localhost:5000/svm';

  constructor(private http: HttpClient) { }

  runSVM(): Observable<any> {
    return this.http.post(`${this.apiUrl}/run`, {});
  }

  getResults(): Observable<any> {
    return this.http.get(`${this.apiUrl}/results`);
  }

  getImage(): Observable<any> {
    return this.http.get(`${this.apiUrl}/image`, { responseType: 'blob' });
  }
}
